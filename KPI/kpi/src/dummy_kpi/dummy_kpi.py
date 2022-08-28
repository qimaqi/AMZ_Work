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

from pprint import pprint

from src.data_handling.data_loader import DataLoader
from src.utils.kpi_utils import bin_cones_by_distance


class DummyKpi:

    def __init__(self, config: dict):
        # We could also be more specific and directly set parameters, e.g.,
        # self.custom_arg = config["custom_arg_1"] if "custom_arg_1" in config else None
        # For this dummy KPI, we are lazy though
        self.config = config
        self.data_loader = DataLoader()

        print('Initialize dummy KPI with this config:')
        pprint(self.config)

    def run(self):
        print('Run dummy KPI')
        pprint(self.config)  # Do something with the config

        # Example of how to get local map
        local_map = self.data_loader.get_topic("/estimation/local_map")

        # The keys correspond to the timestamp in nanoseconds and are ordered chronologically.
        # This is achieved by using an ordered dict.
        print("Timestamps of the first 3 local map cone arrays:")
        for idx, timestamp_ns in enumerate(local_map.keys()):
            print(f'#{idx}: {timestamp_ns}')
            if idx == 2:
                break

        # Look up the output format of cone_arrays by checking the function in
        # data_handling/extractors.py
        # For example, data can be split up as follows.
        print("Some info on the first 3 local map cone arrays:")
        counter = 0
        for timestamp_ns, cone_array in local_map.items():
            # We can bin the cones either by setting the length of a bin
            _ = bin_cones_by_distance(cone_array, bin_length=5)
            # or by directly passing intervals.
            _ = bin_cones_by_distance(cone_array, bins=[0, 5, 10, 15, 20])
            id_cone, prob_type_blue, __, __, __, prob_cone, position_x, __, __, __ , __, __, \
                is_observed = cone_array.T
            print(f'In the {counter}. cone array, the first cone has id {id_cone[0]}, '
                  f'prob_type_blue {prob_type_blue[0]}, is a cone with probability {prob_cone[0]}, '
                  f'has an estimated position in x of {position_x[0]}, andis_observed is set to '
                  f'{is_observed[0]}')
            counter += 1
            if counter == 3:
                break

        # Example of how to get the GTMD array
        gtmd_array = self.data_loader.get_gtmd()
        print("First 3 GTMD cones:")
        print("Format: cone_color_id, latitude, longitude, height, accuracy")
        print(gtmd_array[:3])

    def visualize(self):
        print('Visualize dummy KPI')
        pprint(self.config)  # Do something with the config
