from pyrsistent import l
import rosbag
import numpy as np
import cv2
from cv_bridge import CvBridge
from tqdm import tqdm
from glob import glob
import os
import argparse

# example: "/media/qimaqi/My Passport/tmp/amz/05-28/autocross_2022-05-28-09-10-18.bag"
parser = argparse.ArgumentParser(description='Process the path.')
# parser.add_argument('-p', type=str, metavar='path',
#                     help='starting and ending character')
parser.add_argument('-l', nargs='+', help='Add one or multiple list', required=True)

args = parser.parse_args()

rosbag_path_list = args.l

for rosbag_path in rosbag_path_list:
    self_bag = rosbag.Bag(rosbag_path)
    type_and_topic_info = self_bag.get_type_and_topic_info(
                topic_filters=None)

    topic = '/sensors/forward_camera/image_color'
    DATA_FOLDER = os.path.dirname(rosbag_path)

    topic_name_to_folder_name_dict = {
                "/sensors/fw_lidar/point_cloud_raw": "fw_lidar",
                "/sensors/mrh_lidar/point_cloud_raw": "mrh_lidar",
                "/sensors/forward_camera/image_color": "forward_camera",
                "/perception/lidar/motion_compensated/merge_pc": "merge_pc",
                "/perception/sensor_fusion_overlay/point_cloud_compensated":"compensated_pc"
            }

    pbar = tqdm(total=type_and_topic_info[1][topic].message_count,
                desc=topic)

    data_dir = os.path.join(DATA_FOLDER,
                            topic_name_to_folder_name_dict[topic])
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    counter = 0
    timestamps = []
    images = []
    bridge = CvBridge()

    for _, msg, _ in self_bag.read_messages(topics=[topic]):

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
