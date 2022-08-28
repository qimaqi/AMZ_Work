import numpy as np
import yaml
import os
import cv2
import shutil
from tqdm import tqdm
import sys
from scipy.spatial.transform import Rotation as R
import shutil
from distutils.dir_util import copy_tree
import pathlib

import os

def build_sensor_fusion_2020_structure(rosbag_path,gtmd_file):
    working_dir = os.getcwd()
    rosbag_path_abs = os.path.join(working_dir, rosbag_path) 
    gtmd_file_abs = os.path.join(working_dir, gtmd_file)
    # check the structure exist or not
    root_path_rosbag = os.path.dirname(rosbag_path_abs)
    rosbag_file_name = os.path.basename(rosbag_path_abs)

    root_path_gtmd= os.path.dirname(gtmd_file_abs)
    gtmd_file_name = os.path.basename(gtmd_file_abs)
    src_rosbag_file_path = rosbag_path_abs
    dst_rosbag_file_path = os.path.join(root_path_rosbag,'rosbags',rosbag_file_name)

    src_gtmd_file_path = gtmd_file_abs
    dst_gtmd_file_path = os.path.join(root_path_rosbag,'gtmd',gtmd_file_name)

    home_dir = pathlib.Path.home()
    src_transform_file_path = os.path.join(home_dir, '.amz','static_transformations')
    dst_transform_file_path = os.path.join(root_path_rosbag,'static_transformations')

    if root_path_gtmd == root_path_rosbag:
        print('change file structure to standard one')
        os.mkdir(os.path.join(root_path_rosbag,'rosbags'))
        shutil.move(src_rosbag_file_path,
                            dst_rosbag_file_path)

        os.mkdir(os.path.join(root_path_rosbag,'gtmd'))
        shutil.move(src_gtmd_file_path,
                            dst_gtmd_file_path)
        copy_tree(src_transform_file_path,
                            dst_transform_file_path)
        print('work tree initialize ok')
    
        return root_path_rosbag
    else:
        return os.path.dirname(root_path_rosbag)
            

def build_fiter_2022_structure(data_path):
    ### read incursevily the image and corresponding timestamps
    image_folder = os.path.join(data_path,'forward_camera_filtered')
    ref_timestamps = os.path.join(image_folder,'timestamps.txt')
    imgs_list = os.listdir(image_folder)
    imgs_list.remove('timestamps.txt')
    imgs_list.sort()
    imgs_dst_folder = '/tmp/amz_kpi/forward_camera_filtered'

    fw_lidar_folder = os.path.join(data_path,'fw_lidar_filtered')
    fw_lidar_list = os.listdir(fw_lidar_folder)
    fw_lidar_list.remove('timestamps.txt')
    fw_lidar_list.sort()
    fw_lidar_dst_folder = '/tmp/amz_kpi/fw_lidar_filtered'

    mrh_lidar_folder = os.path.join(data_path,'mrh_lidar_filtered')
    mrh_lidar_list = os.listdir(mrh_lidar_folder)
    mrh_lidar_list.remove('timestamps.txt')
    mrh_lidar_list.sort()
    mrh_lidar_dst_folder = '/tmp/amz_kpi/mrh_lidar_filtered'

    # recursively read line in timestamps


    if (os.path.exists(imgs_dst_folder)):
        shutil.rmtree(imgs_dst_folder)
        os.mkdir(imgs_dst_folder)
    else:
        os.mkdir(imgs_dst_folder)

    if (os.path.exists(fw_lidar_dst_folder)):
        shutil.rmtree(fw_lidar_dst_folder)
        os.mkdir(fw_lidar_dst_folder)
    else:
        os.mkdir(fw_lidar_dst_folder)

    if (os.path.exists(mrh_lidar_dst_folder)):
        shutil.rmtree(mrh_lidar_dst_folder)
        os.mkdir(mrh_lidar_dst_folder)
    else:
        os.mkdir(mrh_lidar_dst_folder)


    with open(ref_timestamps) as file_timestamps:
        lines = file_timestamps.readlines()
        print('****building kpi filter 2022 structure*********')
        time_bar = tqdm(total=len(lines),desc='copying files to tmp')
        for i, timestamps_i in enumerate(lines):
            timestamps_i_str = str(int(float(timestamps_i)*10e8)).zfill(25)
            src_img_i = os.path.join(image_folder,imgs_list[i])
            dst_img_i = os.path.join(imgs_dst_folder,timestamps_i_str+'.png')

            src_fw_lidar_i = os.path.join(fw_lidar_folder,fw_lidar_list[i])
            dst_fw_lidar_i = os.path.join(fw_lidar_dst_folder,timestamps_i_str + '.bin')

            src_mrh_lidar_i = os.path.join(mrh_lidar_folder,mrh_lidar_list[i])
            dst_mrh_lidar_i = os.path.join(mrh_lidar_dst_folder,timestamps_i_str + '.bin')

            ### TODO
            shutil.copy(src_img_i,dst_img_i)
            shutil.copy(src_fw_lidar_i,dst_fw_lidar_i)
            shutil.copy(src_mrh_lidar_i,dst_mrh_lidar_i)
            time_bar.update(1)
        time_bar.close()

            