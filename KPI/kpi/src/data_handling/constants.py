#!/usr/bin/env python3
#
# AMZ Driverless Project
#
# Copyright (c) 2021 Authors:
#   - Stefan Weber <stefwebe@ethz.ch>
#
# All rights reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
#

from enum import Enum

MsgType = Enum('MsgType', 'boundary cone_array gnss pose_2d TFMessage StateDt Image Lidar_points bboxs steering')

DATA_FOLDER = '/tmp/amz_kpi'
# ADD from QI
VIDEO_FOLDER = '/tmp/amz_kpi/kpi_video'
TOPIC_TO_FOLDER_AND_TYPE = {
    '/estimation/boundary': ('boundary', MsgType.boundary),
    '/estimation/local_map': ('local_map', MsgType.cone_array),
    '/estimation/global_map': ('global_map', MsgType.cone_array),
    '/estimation/bounded_path': ('bounded_path', MsgType.boundary),
    '/perception/lidar/cone_array': ('lidar', MsgType.cone_array),
    '/perception/cone_array':('fusion_cone',MsgType.cone_array),
    '/perception/mono_camera/cone_array': ('mono_camera', MsgType.cone_array),
    '/perception/sensor_fusion_2022/cone_array': ('sensor_fusion', MsgType.cone_array),
    '/pilatus_can/GNSS': ('gnss', MsgType.gnss),
    '/tf': ('tf', MsgType.TFMessage),
    '/pilatus_can/steering':('steering',MsgType.steering),
    # TODO check lidar points
    #TODO check why there are two VE topics with different type
    '/pilatus_can/velocity_estimation': ('velocity_estimation', MsgType.StateDt),
    # percpetion used this, but it was the wrong type
    # '/pilatus_can/velocity_estimation': ('velocity', MsgType.pose_2d),
    '/perception/sensor_fusion/cone_array': ('fusion_debug', MsgType.cone_array),
    # try save all images from camera
    # TODO For pilatus save only forward camera bag
}
