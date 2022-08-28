#!/usr/bin/env python3
#
# AMZ Driverless Project
#
# Copyright (c) 2021 Authors:
#   - Stefan Weber <stefwebe@ethz.ch>
#   - Thierry Backes <tbackes@ethz.ch>
#
# All rights reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
#
# pylint: skip-file

import os
import pickle
import numpy as np
import cv2
import tqdm
import sensor_msgs.point_cloud2 as pc2


def extract_boundary(folder, msg):
    # Numpy array columns:
    # boundary_type(0=left_boundary, 1=right_boundary, 2=middle_line), position{x, y, z},
    # confidence
    if not os.path.isdir(folder):
        raise RuntimeError(f"File structure initialization failed, {folder} isn't a directory.")
    timestamp_ns = msg.header.stamp.to_nsec()
    boundary = np.zeros((0, 5))
    for point in msg.left_boundary:
        point_with_conf = np.array([0, point.position.x, point.position.y, point.position.z, point.confidence]).reshape(1, 5)
        boundary = np.vstack((boundary, point_with_conf))
    for point in msg.right_boundary:
        point_with_conf = np.array([1, point.position.x, point.position.y, point.position.z, point.confidence]).reshape(1, 5)
        boundary = np.vstack((boundary, point_with_conf))
    for point in msg.middle_line:
        point_with_conf = np.array([2, point.position.x, point.position.y, point.position.z, point.confidence]).reshape(1, 5)
        boundary = np.vstack((boundary, point_with_conf))

    boundary_path = os.path.join(folder, str(timestamp_ns).zfill(25) + '.npy')
    boundary.tofile(boundary_path)


def extract_cone_array(folder, msg):
    # Numpy array columns:
    # id_cone, prob_type{blue, yellow, orange, orange_big}, prob_cone,
    # position{x, y, z}, position_covariance{x_x, y_y, x_y}, is_observed
    if not os.path.isdir(folder):
        raise RuntimeError(f"File structure initialization failed, {folder} isn't a directory.")

    timestamp_ns = msg.header.stamp.to_nsec()

    # timestamp = float("{}.{}".format(
    # str(msg.header.stamp.secs),
    # str(msg.header.stamp.nsecs).zfill(9)))
    # print('msg.header.stamp.nsecs',msg.header.stamp.nsecs)
    # print('timestamp',timestamp_ns)
    # print('timestamp',timestamp)
    # print('timestamp_ns',timestamp_ns/10e8)

    cone_array = np.zeros((0, 13))
    for cone in msg.cones:
        cone = np.array([
            cone.id_cone, cone.prob_type.blue, cone.prob_type.yellow, cone.prob_type.orange,
            cone.prob_type.orange_big, cone.prob_cone, cone.position.x, cone.position.y,
            cone.position.z, cone.position_covariance.x_x, cone.position_covariance.y_y,
            cone.position_covariance.x_y, cone.is_observed
        ]).reshape(1, 13)
        cone_array = np.vstack((cone_array, cone))

    cone_array_path = os.path.join(folder, str(timestamp_ns).zfill(25) + '.npy')
    cone_array.tofile(cone_array_path)


def extract_gnss(folder, msgs):
    # Numpy array columns:
    # timestamp_ns, RTK_latitude, RTK_longitude, RTK_height, INS_roll, INS_pitch,
    # dual_pitch, dual_heading
    if not os.path.isdir(folder):
        raise RuntimeError(f"File structure initialization failed, {folder} isn't a directory.")
    poses = np.zeros((0, 8))
    for msg in msgs:
        timestamp_ns = msg.header.stamp.to_nsec() - 50e6
        latitude = msg.RTK_latitude
        longitude = msg.RTK_longitude
        height = msg.RTK_height
        ins_roll = msg.INS_roll
        ins_pitch = msg.INS_pitch
        dual_pitch = msg.dual_pitch
        dual_heading = msg.dual_heading
        pose = np.array([timestamp_ns, longitude, latitude, height, ins_roll, ins_pitch, dual_pitch, dual_heading]).reshape(1, 8)
        poses = np.vstack((poses, pose))
    file_path = os.path.join(folder, 'gnss.npy')
    poses.tofile(file_path)
    
    # TODO also save a CSV file in current working dir
    # working_dir = os.getcwd()
    np.savetxt(os.path.join(folder,'gnss.csv'),poses,delimiter=',')

def extract_gnss_csv(gnss_csv_path, folder):
    # Numpy array columns:
    # timestamp_sec_log, RTK_latitude, RTK_longitude, RTK_height
    gnss_data = np.genfromtxt(gnss_csv_path, dtype=None, encoding=None)
    gnss_array = np.zeros((len(gnss_data), 4))
    file_path = os.path.join(folder, 'gnss.npy')
    
    if not os.path.isdir(folder):
        raise RuntimeError(f"File structure initialization failed, {folder} isn't a directory.")
    for idx, gtmd_entry in enumerate(gnss_data):
        for col_idx in range(4):
            gnss_array[idx, col_idx] = gtmd_entry[col_idx]
    gnss_array.tofile(file_path)


def extract_steering(folder, msgs):
    # Numpy array columns:
    # autonomous_msgs/DoubleStamped
    # float64 data
  
    if not os.path.isdir(folder):
        raise RuntimeError(f"File structure initialization failed, {folder} isn't a directory.")
    steerings = np.zeros((0, 2))
    for msg in msgs:
        # print('msg',msg)
        timestamp_ns = msg.header.stamp.to_nsec()
        steering = msg.data

        steering = np.array([timestamp_ns, steering]).reshape(1, 2)
        steerings = np.vstack((steerings, steering))
    file_path = os.path.join(folder, 'steering.npy')
    steerings.tofile(file_path)
    
    # TODO also save a CSV file in current working dir
    # working_dir = os.getcwd()
    np.savetxt(os.path.join(folder,'steering.csv'),steerings,delimiter=',')


def extract_tf(folder, msgs):
    """
    Extracts tf2 messages. Output is a pickled dict where the keys are the different transformations
    between frames. Each transformation has a timestamp, a rotation quaternion and a translation.
    Timestamp is in nanoseconds.
    """

    if not os.path.isdir(folder):
        raise RuntimeError(f"File structure initialization failed, {folder} isn't a directory.")
    transforms_dict = {}
    for msg in msgs:
        for transform in msg.transforms:
            key = transform.child_frame_id + "_to_" + transform.header.frame_id
            transform_data = []
            transform_data.append(transform.header.stamp.secs * 1000000000 + transform.header.stamp.nsecs)
            transform_data.append(transform.transform.translation.x)
            transform_data.append(transform.transform.translation.y)
            transform_data.append(transform.transform.translation.z)
            transform_data.append(transform.transform.rotation.x)
            transform_data.append(transform.transform.rotation.y)
            transform_data.append(transform.transform.rotation.z)
            transform_data.append(transform.transform.rotation.w)
            transforms_dict.setdefault(key, []).append(transform_data)
    file_path = os.path.join(folder, 'tf.pickle')

    with open(file_path, 'wb') as f:
        pickle.dump(transforms_dict, f)


def extract_StateDt(folder, msgs):
    """
    Extracts StateDt message into an array of messages. Each message is a dict with the keys:
    time: float
    vel: [x,y,theta]
    acc: [x,y,theta]
    vel_cov: [x_x,y_y,y_theta]
    """
    data = []
    for msg in msgs:
        state = {}
        state['time'] = msg.header.stamp.secs * 1E9 + msg.header.stamp.nsecs
        state['vel'] = [msg.vel.x, msg.vel.y, msg.vel.theta]
        state['acc'] = [msg.acc.x, msg.acc.y, msg.acc.theta]
        state['vel_cov'] = [msg.vel_cov.x_x, msg.vel_cov.y_y, msg.vel_cov.y_theta]
        data.append(state)

    file_path = os.path.join(folder, 'StateDt.pickle')

    # extract the velocities in the perception format as well
    extract_velocities(folder, msgs)

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def extract_gtmd(gtmd_csv_path, folder):
    # Numpy array columns:
    # Cone_color_id (0=Blue, 1=Yellow, 2=Orange, 3=OrangeBig), Latitude, Longitude,
    # Height_m, Accuracy_m
    gtmd_data = np.genfromtxt(gtmd_csv_path, delimiter=',', dtype=None, encoding=None)
    gtmd_array = np.zeros((len(gtmd_data), 5))
    for idx, gtmd_entry in enumerate(gtmd_data):
        color = gtmd_entry[0]
        if color == 'Blue':
            gtmd_array[idx, 0] = 0
        elif color == 'Yellow':
            gtmd_array[idx, 0] = 1
        elif color == 'Orange':
            gtmd_array[idx, 0] = 2
        elif color == 'OrangeBig':
            gtmd_array[idx, 0] = 3
        else:
            raise RuntimeError(f"Invalid color specified in gtmd file: {color}")
        # Parse Lat, Long, Height, Variance
        for col_idx in range(1, 5):
            gtmd_array[idx, col_idx] = gtmd_entry[col_idx]
    file_path = os.path.join(folder, 'gtmd.npy')
    gtmd_array.tofile(file_path)


def extract_velocities(folder, msgs):
    # Numpy array columns:
    # timestamp_ns, vel_x, vel_y, yaw_rate
    if not os.path.isdir(folder):
        raise RuntimeError(f"File structure initialization failed, {folder} isn't a directory.")
    velocities = np.zeros((0, 4))
    for msg in msgs:
        timestamp_ns = msg.header.stamp.to_nsec()
        vel_x = msg.vel.x
        vel_y = msg.vel.y
        yaw_rate = msg.vel.theta
        velocity = np.array([timestamp_ns, vel_x, vel_y, yaw_rate]).reshape(1, 4)
        velocities = np.vstack((velocities, velocity))
    file_path = os.path.join(folder, 'velocity.npy')
    velocities.tofile(file_path)


def extract_images(folder, msgs):

    if not os.path.isdir(folder):
        raise RuntimeError(f"File structure initialization failed, {folder} isn't a directory.")

    for msg in msgs:
        timestamp_ns = msg.header.stamp.to_nsec()
        im = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        im_name = "images_" + str(timestamp_ns) + ".png"
        save_path_img = os.path.join(folder, im_name)
        cv2.imwrite(save_path_img, im)

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
