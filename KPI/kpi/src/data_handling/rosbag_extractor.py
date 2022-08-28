#!/usr/bin/env python3
#
# AMZ Driverless Project
#
# Copyright (c) 2021 Authors:
#   - Niclas Vödisch <vniclas@ethz.ch>
#   - Stefan Weber <stefwebe@ethz.ch>
#
# All rights reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
#

import os
import shutil
from sqlite3 import Timestamp

import click
from tqdm import tqdm

import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import cv2
from cv_bridge import CvBridge


from . import constants as const
from . import extractors

from src.utils.preprocess_utils_2022 import get_lidar_timestamps, get_camera_timestamps


def convert_msg_to_numpy(pc_msg):
    pc = []
    for point in pc2.read_points(pc_msg, skip_nans=True):
        pc.append(point)
    return np.array(pc, dtype='float64')

def write_point_cloud(file_path, point_cloud):
    _, ext = os.path.splitext(file_path)
    if ext == '.npy':
        np.save(file_path, point_cloud)
    elif ext == '.bin':
        point_cloud.tofile(file_path)
    else:
        print("Saving in specified point cloud format is not possible.")



def extract_lidar_points(folder, msgs):
    if not os.path.isdir(folder):
        raise RuntimeError(f"File structure initialization failed, {folder} isn't a directory.")

    timestamps = []
    pcs = []

    for msg in msgs:
        timestamp_ns = msg.header.stamp.to_nsec()
        pc = convert_msg_to_numpy(msg)
        if pc.size == 0:
                continue    
        file_path = os.path.join(
                folder,
                str(timestamp_ns).zfill(25)) + ".bin"  # zfill
        write_point_cloud(file_path, pc)
        pcs.append(pc)
        timestamps.append(timestamp_ns)
    
    with open(os.path.join(folder, 'timestamps.txt'), 'w') as filehandle:
            filehandle.writelines("{:.6f}\n".format(timestamp)
                                  for timestamp in timestamps)




def extract_data(rosbag_path, gtmd_file,gnss_file, filter_2022):
    print("Extracting data...")
    print(f'rosbag: {rosbag_path}')
    print(f'gtmd:   {gtmd_file}')
    extractor = RosbagExtractor(rosbag_path, gtmd_file,gnss_file, filter_2022)
    extractor.extract_data()


def clean_up():
    print("Deleting rosbag extactor's tmp folder...")
    if os.path.isdir(const.DATA_FOLDER):
        shutil.rmtree(const.DATA_FOLDER)


def init_file_structure():
    if os.path.isdir(const.DATA_FOLDER):
        # Todo(Stefan): Give option in command line to delete this folder
        if click.confirm('The folder exists already do you want to delete it?'):
            click.echo("deleting folder ...")
            shutil.rmtree(const.DATA_FOLDER)
        else:
            raise RuntimeError(f"The directory {const.DATA_FOLDER} exists already. Please delete or rename it.")
    for folder, __ in const.TOPIC_TO_FOLDER_AND_TYPE.values():
        topic_folder_path = os.path.join(const.DATA_FOLDER, folder)
        os.makedirs(topic_folder_path)


# Internal class for extracting rosbags
class RosbagExtractor:

    def __init__(self, rosbag_path='', gtmd_file='',gnss_file='', filter_2022=False):
        self.rosbag_path = rosbag_path
        self.gtmd_file = gtmd_file
        self.gnss_file = gnss_file
        self.buffer = {}
        self.bag = rosbag.Bag(rosbag_path)
        self.type_and_topic_info = self.bag.get_type_and_topic_info(
            topic_filters=None)
        self.filter_2022 = filter_2022
        self.topic_name_to_folder_name_dict = {
            "/sensors/fw_lidar/point_cloud_raw": "fw_lidar",
            "/sensors/mrh_lidar/point_cloud_raw": "mrh_lidar",
            "/sensors/forward_camera/image_color": "forward_camera",
            "/perception/lidar/motion_compensated/merge_pc": "merge_pc",
            "/perception/sensor_fusion_overlay/point_cloud_compensated":"compensated_pc"
        }

    def extract_data(self):
        init_file_structure()
        with rosbag.Bag(self.rosbag_path) as bag:
            with tqdm(desc="Extracting ROS messages", total=bag.get_message_count(list(const.TOPIC_TO_FOLDER_AND_TYPE.keys()))) as pbar:
                for topic, msg, _ in bag.read_messages(const.TOPIC_TO_FOLDER_AND_TYPE.keys()):
                    self.process_msg(topic, msg)
                    pbar.update(1)
        self.write_buffer()
        if self.filter_2022:
            print("Extract lidar and images.")
            self.extract_sensor_msgs_image("/sensors/forward_camera/image_color")
            self.extract_sensor_msgs_point_cloud_2( "/perception/sensor_fusion_overlay/point_cloud_compensated")
            self.extract_sensor_msgs_point_cloud_2( "/sensors/fw_lidar/point_cloud_raw")
            self.extract_sensor_msgs_point_cloud_2( "/sensors/mrh_lidar/point_cloud_raw")
            # do camera-lidar synchorization
            self.camera_lidar_matching()
            

        if not os.path.isfile(self.gtmd_file):
            raise RuntimeError(f"GTMD path {self.gtmd_file} isn't a file.")
       
        extractors.extract_gtmd(self.gtmd_file, const.DATA_FOLDER)
        if self.gnss_file.endswith('.csv'):
            if not os.path.isfile(self.gnss_file):
                raise RuntimeError(f"GNSS path {self.gtmd_file} isn't a file.")
            extractors.extract_gnss_csv(self.gnss_file, const.DATA_FOLDER)
        print("Completed extraction.")

    # Some messages need to be written to disk msg by msg (eg images) due to RAM limitations.
    # Some other messages (eg gnss) are gathered in a buffer and written to a single file, to
    # prevent an unnecessary amount of file loading.
    # Note: Cone Arrays are written msg by msg for simplicity (dynamic matrix size), not due to RAM.
    def process_msg(self, topic, msg):
        folder, data_type = const.TOPIC_TO_FOLDER_AND_TYPE[topic]
        folder_path = os.path.join(const.DATA_FOLDER, folder)
        if data_type is const.MsgType.boundary:
            extractors.extract_boundary(folder_path, msg)
        elif data_type is const.MsgType.cone_array: # filter 2022 will generate new cone array npy
            extractors.extract_cone_array(folder_path, msg)
        elif data_type is const.MsgType.gnss:
            if topic not in self.buffer:
                self.buffer[topic] = [msg]
            else:
                self.buffer[topic].append(msg)
        elif data_type is const.MsgType.TFMessage:
            self.buffer.setdefault(topic, []).append(msg)
        elif data_type is const.MsgType.StateDt:
            self.buffer.setdefault(topic, []).append(msg)
        elif data_type is const.MsgType.pose_2d:
            if topic not in self.buffer:
                self.buffer[topic] = [msg]
            else:
                self.buffer[topic].append(msg)
        elif data_type is const.MsgType.steering:
            self.buffer.setdefault(topic,[]).append(msg)
        else:
            raise NotImplementedError(f"The data type {data_type} isn't handled - yet :).")

    def write_buffer(self):
        for topic in self.buffer:
            folder, data_type = const.TOPIC_TO_FOLDER_AND_TYPE[topic]
            folder_path = os.path.join(const.DATA_FOLDER, folder)
            if data_type is const.MsgType.gnss:
                extractors.extract_gnss(folder_path, self.buffer[topic])
            elif data_type is const.MsgType.TFMessage:
                extractors.extract_tf(folder_path, self.buffer[topic])
            elif data_type is const.MsgType.StateDt:  # extracts in both PER and EST format
                extractors.extract_StateDt(folder_path, self.buffer[topic])
            elif data_type is const.MsgType.pose_2d:  # perception way of storing VE
                extractors.extract_velocities(folder_path, self.buffer[topic])
            elif data_type is const.MsgType.steering:
                extractors.extract_steering(folder_path, self.buffer[topic])
            else:
                raise NotImplementedError(f"The data type {data_type} isn't handled - yet :).")

    def extract_sensor_msgs_point_cloud_2(self, topic):
        pbar = tqdm(total=self.type_and_topic_info[1][topic].message_count,
                    desc=topic)

        data_dir = os.path.join(const.DATA_FOLDER,
                                self.topic_name_to_folder_name_dict[topic])
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        counter = 0
        timestamps = []
        pcs = []
        for _, msg, time in self.bag.read_messages(topics=[topic]):
            # Weird issue is that time and msg.header.stamp differ (at around 0.1s)
            # Taking msg.header.stamp for now

            pbar.update(1)
            timestamp_ns = msg.header.stamp.to_nsec()

            timestamp = float("{}.{}".format(
                str(msg.header.stamp.secs),
                str(msg.header.stamp.nsecs).zfill(9)))

            pc = convert_msg_to_numpy(msg)
            if pc.size == 0:
                continue
            # os.path.join(folder, str(timestamp_ns).zfill(25) + '.npy')
            file_path = os.path.join(
                data_dir,
                str(timestamp_ns).zfill(25)) + ".bin" 
            write_point_cloud(file_path, pc)

            pcs.append(pc)
            timestamps.append(timestamp)

            counter += 1

        pbar.close()

        with open(os.path.join(data_dir, 'timestamps.txt'), 'w') as filehandle:
            filehandle.writelines("{:.6f}\n".format(timestamp)
                                  for timestamp in timestamps)

        return pcs, timestamps

    def extract_sensor_msgs_image(self, topic):
        pbar = tqdm(total=self.type_and_topic_info[1][topic].message_count,
                    desc=topic)

        data_dir = os.path.join(const.DATA_FOLDER,
                                self.topic_name_to_folder_name_dict[topic])
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        counter = 0
        timestamps = []
        images = []
        bridge = CvBridge()

        for _, msg, _ in self.bag.read_messages(topics=[topic]):

            timestamp = float("{}.{}".format(
                str(msg.header.stamp.secs),
                str(msg.header.stamp.nsecs).zfill(9)))
            
            timestamp_ns = msg.header.stamp.to_nsec()

            image = bridge.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imwrite(os.path.join(data_dir,
                                     str(timestamp_ns).zfill(25) + '.png'), image)
            timestamps.append(timestamp)
            images.append(image)

            counter += 1

            pbar.update(1)

        pbar.close()

        with open(os.path.join(data_dir, 'timestamps.txt'), 'w') as filehandle:
            filehandle.writelines("{:.6f}\n".format(timestamp)
                                  for timestamp in timestamps)

        return images, timestamps

    def camera_lidar_matching(self):
        """
        This function choose camera as basic data and match the nearest lidar point to camera frame.
        Results is new data folder named matched_lidar
        """

        timestamp_lidar = get_lidar_timestamps(const.DATA_FOLDER)
        timestamp_camera = get_camera_timestamps(const.DATA_FOLDER)
        camera_ref_timestamps = timestamp_camera['forward_camera']
        
        induces_dict = {}
        for key in timestamp_lidar.keys():
            induces_dict[key] = []

        for key, pc_timestamps_lidar in timestamp_lidar.items():
            print('key', key)
            for ref_timestamp in camera_ref_timestamps:
                time_diff = np.abs(pc_timestamps_lidar- ref_timestamp) # +0.05 
                min_idx = time_diff.argmin()

                if time_diff[min_idx] > 0.05:
                    induces_dict[key].append(-1)
                else:
                    induces_dict[key].append(min_idx)

            
        self.filter_point_cloud(induces_dict)
    
    def filter_point_cloud(self, induces_dict):
        print('matching images and lidar points')
        for key in induces_dict:
            src_point_cloud_folder_path = os.path.join(const.DATA_FOLDER, key)  # TODO
            dst_point_cloud_folder_path = os.path.join(const.DATA_FOLDER,'matched_' + key)
            src_camera_folder_path = os.path.join(const.DATA_FOLDER,'forward_camera')

            if os.path.exists(dst_point_cloud_folder_path):
                print(
                    "The folder {} exist already indicating that the data has already been matched!"
                    .format(dst_point_cloud_folder_path))
                print("{} will be removed and the data will be rematched.".
                        format(dst_point_cloud_folder_path))
                shutil.rmtree(dst_point_cloud_folder_path)
            os.makedirs(dst_point_cloud_folder_path)

            # Get all files in a list and remove timestamp.txt
            filenames = []
            for (_, _,
                    current_filenames) in os.walk(src_point_cloud_folder_path):
                filenames.extend(current_filenames)
                break
            filenames.remove("timestamps.txt")

            filenames_imgs = []
            for (_, _,
                    current_filenames) in os.walk(src_camera_folder_path):
                filenames_imgs.extend(current_filenames)
                break
            filenames_imgs.remove("timestamps.txt")

            _, extension = os.path.splitext(filenames[0])

            filenames.sort()
            filenames_imgs.sort()
            # print("filename", filenames)
            # print('induces_dict',induces_dict)
            # for point_cloud_idx in induces_dict:
            
            pbar = tqdm(total=len(induces_dict[key]), desc='lidar')

            for idx, point_cloud_idx in enumerate(induces_dict[key]):
                if point_cloud_idx != -1:
                    src_point_cloud_file_path = os.path.join(
                        src_point_cloud_folder_path, filenames[point_cloud_idx])
                    dst_point_cloud_file_path = os.path.join(
                        dst_point_cloud_folder_path, filenames_imgs[idx][:-3] + 'bin')
                    shutil.copy(src_point_cloud_file_path,
                                dst_point_cloud_file_path)
                    ### TODO add ego motion compensation or directly use compensated point cloud
                    
                else: 
                    # create empty point file
                    dst_point_cloud_file_path = os.path.join(
                        dst_point_cloud_folder_path, filenames_imgs[idx][:-3] + 'bin')
                    
                    with open(dst_point_cloud_file_path, 'w'):
                        pass
                pbar.update(1)
            pbar.close()