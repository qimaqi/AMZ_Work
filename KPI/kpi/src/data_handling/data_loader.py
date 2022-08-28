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

import os

from . import constants as const
from . import loaders


class DataLoader:

    def __init__(self):
        self.buffer = {}

    def get_topic(self, topic):
        if topic in self.buffer:
            return self.buffer[topic]

        if topic not in const.TOPIC_TO_FOLDER_AND_TYPE:
            raise RuntimeError(f"Requested topic {topic}, which isn't configured.")
        folder, data_type = const.TOPIC_TO_FOLDER_AND_TYPE[topic]
        folder_path = os.path.join(const.DATA_FOLDER, folder)

        if data_type is const.MsgType.boundary:
            result = loaders.load_boundaries(folder_path)
        elif data_type is const.MsgType.cone_array:
            result = loaders.load_cone_arrays(folder_path)
        elif data_type is const.MsgType.gnss:
            result = loaders.load_gnss(folder_path)
        elif data_type is const.MsgType.TFMessage:
            result = loaders.load_tf(folder_path)
        elif data_type is const.MsgType.StateDt:  # VE message
            result = loaders.load_StateDt(folder_path)
        elif data_type is const.MsgType.pose_2d:  # VE message
            result = loaders.load_velocities(folder_path)
        else:
            raise RuntimeError(f"Data type {data_type} is not configured.")
        self.buffer[topic] = result
        return result

    def get_gtmd(self):
        if 'gtmd' in self.buffer:
            return self.buffer['gtmd']
        gtmd_array = loaders.load_gtmd(const.DATA_FOLDER)
        self.buffer['gtmd'] = gtmd_array
        return gtmd_array

    def align_gnss(self):
        folder, data_type = const.TOPIC_TO_FOLDER_AND_TYPE[topic]
        folder_path = os.path.join(const.DATA_FOLDER, folder)



        result = loaders.load_gnss(folder_path)

