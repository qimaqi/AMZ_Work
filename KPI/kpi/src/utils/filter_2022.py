from ctypes import create_string_buffer
from operator import matmul
from os import stat
import glob
from unittest.util import _count_diff_all_purpose
import numpy as np
import cv2, yaml
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import statistics 
import os
from scipy.stats import kde
import math
from src.data_handling import constants as const

import time
from tqdm import tqdm
from collections import OrderedDict
import shutil
import pathlib

class filtering_2022:
    def __init__(self, conf_thres, lidar_type, filtering_tunning=0 ,image_output_flag=True, heat_output_flag=True, save_result=False):
        self.conf_thres = conf_thres
        self.images_folder = os.path.join(const.DATA_FOLDER, 'forward_camera') #filtered' 
        self.lidar_type = lidar_type
        if self.lidar_type == 'fw_lidar':
            self.lidar_folder = os.path.join(const.DATA_FOLDER, 'fw_lidar_filtered')   ### TODO
        elif self.lidar_type == 'mrh_lidar':
            self.lidar_folder = os.path.join(const.DATA_FOLDER, 'mrh_lidar_filtered')   ### TODO
        elif self.lidar_type == 'compensated_pc':
            self.lidar_folder = os.path.join(const.DATA_FOLDER, 'matched_compensated_pc')   ### TODO   

        self.boxes_folder = os.path.join(const.DATA_FOLDER, 'forward_labels', 'yolo_output','labels') 
        self.output_folder = os.path.join(const.DATA_FOLDER, 'filter_result') 
        self.save_cone_array_folder = os.path.join(const.DATA_FOLDER, 'sensor_fusion')
        home_path = pathlib.Path.home()
        self.static_transforms = os.path.join(home_path,'.amz','static_transformations')
        self.intrinsic_file = os.path.join(home_path,'.amz','static_transformations','forward.yaml')
        self.save_result = save_result

        if (os.path.exists(self.output_folder)):
            shutil.rmtree(self.output_folder)
            os.mkdir(self.output_folder)
        else:
            os.mkdir(self.output_folder)

        if (os.path.exists(self.save_cone_array_folder)):
            shutil.rmtree(self.save_cone_array_folder)
            os.mkdir(self.save_cone_array_folder)
        else:
            os.mkdir(self.save_cone_array_folder)

        ### read the intrinsic extrinsic
        # self.extrinsics_mrh2fc = self.extrinsics_load('extrinsics_mrh_forward')
        self.extrinsics_fw2fc = self.extrinsics_load('extrinsics_fw_forward') # extrinsics_fw_forward
        self.mrh_lidar_to_egomotion, self.fw_lidar_to_mrh_lidar = self.extrinsics_load('static_transformations')
        self.extrinsics_mrh2fc  = np.matmul(self.extrinsics_fw2fc, self.transfromInverse(self.fw_lidar_to_mrh_lidar))
        
        self.intrinsics_fc, self.intrinsics_fc_D = self.intrinsics_load('forward')
        self.fw_camera_to_egomotion = self.get_fw_cam_to_egomotion()
        self.fw_lidar_to_egomotion = self.extrinsics_load('extrinsics_fw_egomotion')
    ### self defin function

    def load_data(self, img_file, pc_file, bbox_file):
        img = cv2.imread(img_file)
        pc = np.fromfile(pc_file).reshape(-1, 6) # X, Y, Z, Intensity, Timestamp, Channel
        
        f = open(bbox_file)
        raw = f.read()
        f.close()
        boxes = [[float(i) for i in k.split(" ") if not i==""] for k in raw.split("\n") if k] #list of boxes: Class, Depth, X, Y (of center), W, H
        return img, pc, boxes

    def intrinsics_load(self, file):
        intrinsics_file = os.path.join(self.static_transforms, file +'.yaml')
        with open(intrinsics_file, 'r') as f:
            fw_cam_trans = yaml.safe_load(f)
            camera_mtx = np.array(fw_cam_trans['camera_matrix']['data']).reshape(3, 3)
            distCoeffs = np.array(fw_cam_trans['distortion_coefficients']['data'])
        return camera_mtx, distCoeffs

    
    def extrinsics_load(self, file):
        if file in ['extrinsics_mrh_forward','extrinsics_mrh_left','extrinsics_mrh_right', 'extrinsics_fw_forward','extrinsics_fw_egomotion']:
            f = cv2.FileStorage(os.path.join(self.static_transforms, file +'.yaml'), cv2.FILE_STORAGE_READ)
            r = f.getNode("R_mtx").mat()
            t = f.getNode("t_mtx").mat()
            return np.vstack((np.hstack((r, t)), np.array([0.0, 0.0, 0.0, 1.0]))) #converting matrix to 4x4 
        elif file == 'static_transformations':
            with open(os.path.join(self.static_transforms,file + '.yaml')) as f:
                stats = yaml.safe_load(f)
                r1 = stats['2020-08-23_tuggen']['mrh_lidar_to_egomotion']['rotation']
                t1 = stats['2020-08-23_tuggen']['mrh_lidar_to_egomotion']['translation']
                r2 = stats['2020-08-23_tuggen']['fw_lidar_to_mrh_lidar']['rotation']
                t2 = stats['2020-08-23_tuggen']['fw_lidar_to_mrh_lidar']['translation']
            rot1 = R.from_euler('zyx', np.array([r1['yaw'], r1['pitch'], r1['roll']]), degrees=False).as_matrix()
            rot2 = R.from_euler('zyx', np.array([r2['yaw'], r2['pitch'], r2['roll']]), degrees=False).as_matrix()
            mrh_lidar_to_egomotion = np.column_stack((rot1, np.array([t1['x'], t1['y'], t1['z']])))
            fw_lidar_to_mrh_lidar = np.column_stack((rot2, np.array([t2['x'], t2['y'], t2['z']])))
            mrh_lidar_to_egomotion = np.vstack((mrh_lidar_to_egomotion, np.array([0.0, 0.0, 0.0, 1.0])))
            fw_lidar_to_mrh_lidar = np.vstack((fw_lidar_to_mrh_lidar, np.array([0.0, 0.0, 0.0, 1.0])))

            return mrh_lidar_to_egomotion, fw_lidar_to_mrh_lidar

    def get_cone_array(self):
        # read boxes file first
        image_output_folder = os.path.join(self.output_folder, 'image')
        if not (os.path.exists(image_output_folder)):
            os.mkdir(image_output_folder)
        box_output_folder = os.path.join(self.output_folder,'boxes' )
        if not (os.path.exists(box_output_folder)):
            os.mkdir(box_output_folder)
        
        pbar = tqdm(total=len(os.listdir(self.boxes_folder )), # [3000:]
                    desc='images')
        for box_file in os.listdir(self.boxes_folder): # [3000:]
            if box_file.endswith('.txt'):
                timestamp_str = os.path.splitext(box_file)[0]
                cone_array = np.zeros((0, 13))
                image_file = os.path.join(self.images_folder,timestamp_str + '.png')
                lidar_file = os.path.join(self.lidar_folder, timestamp_str+'.bin')
                box_file = os.path.join(self.boxes_folder, timestamp_str + '.txt')
                img_i, pc_i, boxes_i = self.load_data(image_file, lidar_file, box_file)
                # "data/static_transformations/" + cam + ".yaml"

                img_i, img_with_pc, img_ext, intrinsics_inv = self.project_points(img_i, pc_i)


                cones_i = self.median_filtering(timestamp_str, img_i, img_with_pc, img_ext, boxes_i, 
                                                                        intrinsics_inv, image_output_folder, box_output_folder) 
                for cone_dict in cones_i:
                    cone = np.array([
                        cone_dict['id_cone'], cone_dict['blue_prob'],cone_dict['yellow_prob'],cone_dict['orange_prob'],cone_dict['big_orange_prob'], 
                        cone_dict['prob_cone'], cone_dict['x'], cone_dict['y'],cone_dict['z'],cone_dict['x_x_position_covariance'],cone_dict['y_y_position_covariance'] ,
                        cone_dict['x_y_position_covariance'],cone_dict['is_observed'] 
                    ]).reshape(1, 13)
                    cone_array = np.vstack((cone_array, cone))
                
                cone_array_path = os.path.join(self.save_cone_array_folder, str(timestamp_str).zfill(25) + '.npy')
                cone_array.tofile(cone_array_path)
                pbar.update(1)
        pbar.close()

    def project_points(self, img, pc):
        if self.lidar_type == 'mrh_lidar':
            extrinsics = self.extrinsics_mrh2fc
        elif self.lidar_type == 'fw_lidar':
            extrinsics = self.extrinsics_fw2fc
        else: 
            extrinsics = self.extrinsics_mrh2fc

        
        intrinsics = np.float64(self.intrinsics_fc)
        fw_distCoeffs = np.float64(self.intrinsics_fc_D)
        intrinsics_inv = np.linalg.inv(intrinsics)

        points = pc[:,:3].T # Extracting X, Y, Z in Lidar Frame

        src_point = np.float64(np.copy(points.T))
        rotation_mat = np.float64(extrinsics[:3,:3])
        trans_vec = np.float64(extrinsics[:3,3])
        rotation_vector, _ = cv2.Rodrigues(rotation_mat)

        # print('src_point',len(src_point))
        # print('rotation_mat',rotation_mat)
        # print('trans_vec',trans_vec)
        # print('rotation_vector',rotation_vector)
        # print('fw_distCoeffs',fw_distCoeffs)
        # print('intrinsics',intrinsics)


        points = np.vstack((points, np.ones((1, points.shape[1]))))
        
        points_cam = np.matmul(extrinsics[:3,:], points) #Note: index 2 is the depth of the lidar points from the camera frame
        ## Question: Check with Gowtham in case we should use the norm of the planar coordinates instead of just one dimension for the depth
        points = np.matmul(intrinsics, points_cam)
        #### ! no distortion!

        # 
        points[:2, :] /= points[2, :]

        # Adding red lidar points on the image
        if len(src_point) != 0:
            result_points = cv2.projectPoints(src_point, rotation_vector, trans_vec, cameraMatrix=intrinsics, distCoeffs=fw_distCoeffs)
            target_points = np.array(result_points[0]).reshape(-1,2)
            target_points = target_points.T
            points = target_points

        img_with_pc = img.copy()
        for i in range(points.shape[1]):
            try:
                cv2.circle(img_with_pc, (int(np.round(points[0, i])), int(np.round(points[1, i]))),
                        2, color=255, thickness=-1)
            except:
                pass

        #Points (X, Y, Depth) in camera frame
        if len(src_point) != 0:
            points = np.vstack((points, points_cam[2,:]))
        else:
            points[2,:] = points_cam[2,:]
        points_ext = points.T
        # print('points_ext',points_ext)

        (H, W, _) = img.shape
        # print("Image Width:", W,", Image Height:", H)
        img_ext = np.zeros((H, W, 4),dtype = np.float32)
        img_ext[:,:,:3] = img/255
    
        for i in range(points_ext.shape[0]):
            # lost the track of lidar points information
            if (points_ext[i,1] > 0 and points_ext[i,1] < H-1 and points_ext[i,0] > 0 and points_ext[i,0] < W-1):
                img_ext[int(np.round(points_ext[i,1])), int(np.round(points_ext[i,0])),3] = points_ext[i,2]

        return img, img_with_pc, img_ext, intrinsics_inv

    def median_filtering(self, img_index, img, img_with_pc, img_ext, boxes, intrinsics_inv, image_output_folder, box_output_folder):
        conf_thres = self.conf_thres
        (H, W, _) = img.shape
        cones_i = []
        for b in range(len(boxes)):
            #box_output_file = 'box_'+ str(b) + '.png'
            # (box_classes, camera_box_depth, box_x_center, box_y_center, box_width, box_height) = boxes[b][:]
            ### standard coco output no depth
            (box_classes, box_x_center, box_y_center, box_width, box_height) = boxes[b][:]
            (box_x_center, box_y_center, box_width, box_height) = (int(np.round(W*box_x_center)), int(np.round(H*box_y_center)), int(np.round(W*box_width)), int(np.round(H*box_height)))
        
            img_crop = img_ext[box_y_center-int(box_height/2):box_y_center+int(box_height/2), box_x_center-int(box_width/2):box_x_center+int(box_width/2),:]
            (H_c, W_c, _) = img_crop.shape
            #print("BBox Res:",H_c,"x", W_c)
            img_crop_copy = img_crop[:,:,:3].astype(np.float32)
            count = 0
            countFiltered = 0
            depths = []
            for x in range(W_c):
                for y in range(H_c):
                    pixelDistance = img_crop[y][x][3]
                    if (pixelDistance != 0):
                        count += 1
                        #print("Lidar Point {:d}Hx{:d}W, Depth:{:f}".format(y,x,img_crop[y,x,3]))
                        #print("   Lidar Point Percent {:f}Hx{:f}W, Depth:{:f}".format(y/H_c,x/W_c,img_crop[y,x,3]))
                        
                        if ((x/W_c)>0.20 and (x/W_c)<0.80 and (y/H_c)>0.05 and (y/H_c)<0.95):
                            # turning filtered points green and storing their depth
                            if self.save_result:
                                cv2.circle(img_crop_copy,(x,y),0, color=(0,255,0), thickness=-1)
                                cv2.circle(img_with_pc,(box_x_center-int(box_width/2)+x,box_y_center-int(box_height/2)+y),2, color=(0,255,0), thickness=-1)
                            depths.append(img_crop[y,x,3])
                            countFiltered += 1
                        else:
                            if self.save_result:
                                cv2.circle(img_crop_copy,(x,y),0, color=(255,0,0), thickness=-1)
            #Taking the median of the filtered points as the distance
            if (len(depths) != 0):
                median_filtered_depth = statistics.median(depths) 
                cone_x, cone_y, cone_z = self.get_cone_in_egomotion(box_x_center, box_y_center, median_filtered_depth, intrinsics_inv)

                # cones_x.append(cone_x)
                # cones_y.append(cone_y)
                # cones_z.append(cone_z)
                cone_dict = {}
                if box_classes == 0:
                    cone_dict['blue_prob'] = 0.8
                    cone_dict['yellow_prob'] = 0.1
                    cone_dict['orange_prob'] = 0.1
                    cone_dict['big_orange_prob'] = 0
                if box_classes == 1:
                    cone_dict['blue_prob'] = 0.1
                    cone_dict['yellow_prob'] = 0.8
                    cone_dict['orange_prob'] = 0.1
                    cone_dict['big_orange_prob'] = 0
                else:
                    cone_dict['blue_prob'] = 0
                    cone_dict['yellow_prob'] = 0
                    cone_dict['orange_prob'] = 0
                    cone_dict['big_orange_prob'] = 0  

                cone_dict['x_x_position_covariance'] = 0.3 
                cone_dict['y_y_position_covariance'] = 0.3     
                cone_dict['x_y_position_covariance'] = 0.1        

                cone_dict['id_cone'] = b
                cone_dict['prob_cone'] = conf_thres
                cone_dict['x'] = cone_x
                cone_dict['y'] = cone_y
                cone_dict['z'] = cone_z
                cone_dict['is_observed'] = True

                # print('final prediction of cone x y z',cone_x, cone_y, cone_z )
                
                cones_i.append(cone_dict)
                
                box_output_file = 'box'+ str(b) + '_predDistInCameraFr_' + str(round(median_filtered_depth, 2)) + 'm_predDistInEgoFr_'+ str(round(math.sqrt(cone_x**2 + cone_y**2), 2)) +'m.png'
                #box_output_file = 'box'+ str(b) + '_predDist_' + str(round(median_filtered_depth, 2)) + 'm.png'
            else:
                median_filtered_depth = 0
                box_output_file = 'box'+ str(b) + '_predDist_NONE.png'

            # if self.save_result:
            #     plt.rcParams['figure.dpi'] = 300
            #     plt.rcParams['savefig.dpi'] = 300
            #     plt.imshow(img_crop_copy[:,:,:3].astype('uint8'))
            #     plt.yticks([])
            #     plt.xticks([])
            #     plt.show()
            #     plt.savefig(os.path.join(box_output_folder, box_output_file))

            # print("Total Number of LiDAR Points:",count)
            # print("Filtered LiDAR Points:", countFiltered)
            # print("Median Filtered Depth:", median_filtered_depth,"out of", len(depths), "points")
            # print("Camera GT Distnace:",camera_box_depth)

        if self.save_result:   
            image_output_file = 'image'+ str(img_index) +'.png'
            # plt.rcParams['figure.dpi'] = 300
            # plt.rcParams['savefig.dpi'] = 300
            plt.imshow(img_with_pc.astype('uint8'))
            plt.yticks([])
            plt.xticks([])
            plt.show()
            plt.savefig(os.path.join(image_output_folder, image_output_file))

            plt.close('all')

        return cones_i

    def get_cone_in_egomotion(self, box_x_center, box_y_center, filtered_depth, intrinsics_inv):
        
        fw_cam_to_egomotion = self.get_fw_cam_to_egomotion()
        #print(box_x_center, box_y_center)
        #box_pixels = np.array([filtered_depth*box_x_center, filtered_depth*box_y_center, filtered_depth*1]).reshape(-1,1)
        box_pixels = np.array([box_x_center, box_y_center, 1]).reshape(-1,1)
        cone_camera = np.matmul(intrinsics_inv, box_pixels)
        # print('cone_camera',cone_camera)

        cone_camera = cone_camera * filtered_depth
        cone_camera = np.vstack((cone_camera, np.ones((1,1))))
        #print(cone_camera)
        cone_egomotion = np.matmul(fw_cam_to_egomotion[:3,:], cone_camera) 
        #print(cone_egomotion) # Something is wrong here, most probably with the projection  
        
        cone_x = cone_egomotion[0, 0]
        cone_y = cone_egomotion[1, 0]
        cone_z = cone_egomotion[2, 0]
        # print('cone_x, cone_y, cone_z',cone_x, cone_y, cone_z)
        return cone_x, cone_y, cone_z

    def get_fw_cam_to_egomotion(self):        
        mrh_lidar_to_fw_camera = self.extrinsics_mrh2fc
        r = mrh_lidar_to_fw_camera[:3,:3]
        t = mrh_lidar_to_fw_camera[:3,3]
        r_new = r.T
        t_new = - np.matmul(r_new, t)
        fw_cam_to_mrh_lidar = np.column_stack((r_new, t_new))
        fw_cam_to_mrh_lidar = np.vstack((fw_cam_to_mrh_lidar, np.array([0.0, 0.0, 0.0, 1.0]))) #converting matrix to 4x4

        
        fw_camera_to_egomotion = np.matmul(self.mrh_lidar_to_egomotion, fw_cam_to_mrh_lidar) #4x4 matrix

        return fw_camera_to_egomotion

    def get_fw_lidar_to_fw_cam(self):
        mrh_lidar_to_fw_camera = self.extrinsics_mrh2fc
        x = 0.05
        y = -2.0
        z = -0.69
        yaw = 3.139
        pitch = -3.13
        roll = 3.11 
        rot = R.from_euler('zyx', np.array([yaw, pitch, roll]), degrees=False).as_matrix()
        fw_lidar_to_mrh_lidar = np.column_stack((rot, np.array([x, y, z])))           
        fw_lidar_to_mrh_lidar = np.vstack((fw_lidar_to_mrh_lidar, np.array([0.0, 0.0, 0.0, 1.0]))) #converting matrix to 4x4 
        fw_lidar_to_fw_camera = np.matmul(mrh_lidar_to_fw_camera, fw_lidar_to_mrh_lidar)
        # print('fw_lidar_to_fw_camera',fw_lidar_to_fw_camera)
        return fw_lidar_to_fw_camera

    def transfromInverse(self,T):
        # inv(T) = [inv(R) | -inv(R)*t]
        inv_T = np.zeros_like(T) # 4x4
        inv_T[3,3] = 1.0
        inv_T[:3,:3] = np.transpose(T[:3,:3])
        inv_T[:3,3] = np.dot(-np.transpose(T[:3,:3]), T[:3,3])
        return inv_T


# filter_instance = filtering_2022(conf_thres = 0.8, save_result = True, lidar_type='compensated_pc')
# filter_instance.get_cone_array()