#!/usr/bin/env python3
#
# AMZ Driverless Project
#
# Copyright (c) 2021 Authors:
#   - Thierry Backes <tbackes@ethz.ch>
#
# All rights reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
#

# pylint: disable=unused-wildcard-import, method-hidden
# pylint: enable=too-many-lines
import math
import io
from typing import Optional, List, Tuple
import warnings

import numpy as np

from plotnine import *
from plotnine.data import *
import pandas as pd

from src.utils import kpi_utils

warnings.filterwarnings('ignore')


def cones_to_worldframe(cones: np.ndarray, gnss_offset: np.ndarray, heading_offset: float, gnss_null: np.ndarray, heading_null: float) -> np.ndarray:
    """
    Transform cones from egomotion into world frame using RTK. This is more precise than using
    tf2 egomotion to world which is based on velocity estimation. This involves 2 steps:
    First transform from egomotion back into RTK frame. Then from the RTK frame transform into
    world frame where (0,0) is at the car
    """

    # check if cones even exist
    if cones.size == 0:
        # return not iterable array
        return np.zeros((0, 3))
    cones_no_color = cones[:, 1:]  # ignore color and add it back later

    # cone array might have x,y,z coordinates. We're too cheap to work in 3D so just remove it
    if cones_no_color.shape[1] == 3:
        cones_no_color = cones_no_color[:, :-1]

    # Compute from egomotion to RTK offset
    cones_no_color[:, 0] -= kpi_utils.RTK_COG_OFFSET

    # Transform from egomotion back into world frame
    heading_offset = (heading_offset + 90) * math.pi / 180.0

    rotation_matrix = np.array([[math.cos(heading_offset), -math.sin(heading_offset)], [math.sin(heading_offset), math.cos(heading_offset)]])

    # note the multiplication is with rotation_matrix.T, the inverse rotation
    cones_rtk_frame = np.matmul(rotation_matrix.T, cones_no_color.T).T
    cones_rtk_frame += gnss_offset

    # The cones are in the RTK/GTMD frame which is in epsg:4326 and (0,0) is somewhere in the ocean.
    # Next compute their location relative to the starting position of the car so that (0,0) is where the car is
    # We call this our world-frame
    # First translate, then rotate. Otherwise the rotation is around the ocean and not the car
    cones_rtk_frame -= gnss_null
    heading_null = (heading_null + 90) * math.pi / 180.0
    transformation_matrix = np.array([[math.cos(heading_null), -math.sin(heading_null)], [math.sin(heading_null), math.cos(heading_null)]])
    cones_world_frame = np.matmul(transformation_matrix, cones_rtk_frame.T).T

    return np.column_stack((cones[:, 0], cones_world_frame))  # add color back

def cones_to_worldframe_tf(cones: np.ndarray,tmat, gnss_null: np.ndarray, heading_null: float) -> np.ndarray:
    """
    Transform cones from egomotion into world frame using RTK. This is more precise than using
    tf2 egomotion to world which is based on velocity estimation. This involves 2 steps:
    First transform from egomotion back into RTK frame. Then from the RTK frame transform into
    world frame where (0,0) is at the car
    try to use tf
    """

    # check if cones even exist
    if cones.size == 0:
        # return not iterable array
        return np.zeros((0, 3))
    cones_no_color = cones[:, 1:]  # ignore color and add it back later

    # cone array might have x,y,z coordinates. We're too cheap to work in 3D so just remove it
    if cones_no_color.shape[1] == 3:
        cones_no_color = cones_no_color[:, :-1]

    cone_pos_ego = cones_no_color
    # homography
    # add z channel

    cols = np.shape(cone_pos_ego)[0]
    cone_pos_ego = np.append(cone_pos_ego, np.zeros((cols,1)),axis=1)
    # cone_pos_world = np.append(cone_pos_world, np.ones((cols,1)),axis=1)

    # change world to egomotion
    rot_mat = tmat[:3,:3]
    # print('rot_mat',rot_mat)
    # print('rot_mat',np.shape(cone_pos_ego.T))
    # print('tmat[:,:3]',np.shape(tmat[:3,3]))
    heading_null = (heading_null + 90) * math.pi / 180.0
    transformation_matrix = np.array([[math.cos(heading_null), -math.sin(heading_null)], [math.sin(heading_null), math.cos(heading_null)]])
    trans = tmat[:3,3]
    # print('trans',trans)

    cone_pos_world = np.matmul(rot_mat, cone_pos_ego.T).T
    cone_pos_world += trans
    cone_pos_ego_2d = cone_pos_ego[:,:2]
    cones_world_frame = np.matmul(transformation_matrix, cone_pos_ego_2d.T).T

    return np.column_stack((cones[:,0], cones_world_frame))    # add color back

def plot_track_ggplot(data,
                      all_gtmd_cones: List[List[float]],
                      timestamp: int,
                      bucket: Tuple[int, int],
                      matching_distance: float,
                      create_video: bool,
                      title: str,
                      current_frame: int,
                      velocity,
                      gnss_offset: np.ndarray,
                      heading_offset: float,
                      gnss_null: np.ndarray,
                      heading_null: float,
                      tmat,
                      layout: int = 0) -> Optional[io.BytesIO]:
    # pylint: disable=too-many-statements,too-many-locals,too-many-branches,too-many-arguments
    """
    Plots the track. Either shows the track plot or returns it in a BytesIO buffer to be used with ffmpeg to
    create a video.

    First this function extracts the required data from the arguments. Then it transforms the cones from
    egomotion into worldframe. The next step is to plot the data and the last step is to add a legend
    and statistics to the plot.
    """

    ## Extract data

    gtmd_cones_dict = data['gtmd_cones_dict']
    gtmd_cones = gtmd_cones_dict['pos_with_color']
    cone_dict = data['cone_dict']
    cones = cone_dict['pos_with_color']
    unmatched_gtmd_idx = data['unmatched_gtmd_idx']
    unmatched_cones_idx = data['unmatched_cones_idx']
    multiple_matched_gtmd_idx = data['multiple_matched_gtmd_idx']
    matched_cones_idx = data['matched_cones_idx']
    correct_matches = data['correct_matches']
    gtmd_no_match = data['gtmd_no_match']
    gtmd_multiple_match = data['gtmd_multiple_match']
    unmatched_cones = data['unmatched_cones']
    cone_match_wrong_color = data['cone_match_wrong_color']
    unmatched_gtmd_counter = data['unmatched_gtmd_counter']
    multiple_matched_gtmd_counter = data['multiple_matched_gtmd_counter']
    no_matched_cones_counter = data['no_matched_cones_counter']

    if len(cones) > 0:
        blue_cones_count = (cones[:, 0] == 0).sum()
        yellow_cones_count = (cones[:, 0] == 1).sum()
    else:
        blue_cones_count = 0
        yellow_cones_count = 0
    if len(gtmd_cones) > 0:
        blue_gtmd_count = (gtmd_cones[:, 0] == 0).sum()
        yellow_gtmd_count = (gtmd_cones[:, 0] == 1).sum()
    else:
        blue_gtmd_count = 0
        yellow_gtmd_count = 0

    all_gtmd_cones_dict = kpi_utils.unpack_gtmd_cones(all_gtmd_cones)

    ## Transform data

    # transform (grey) GTMD cones that stay fixed. Always use timestamp 1
    # gtmd_fixed = cones_to_worldframe(all_gtmd_cones_dict['pos_with_color'], gnss_offset, heading_offset, gnss_null, heading_null)
    # Find closest timestamp in ego to world transformations
    # new_gtmd_cones = cones_to_worldframe(gtmd_cones, gnss_offset, heading_offset, gnss_null, heading_null)
    # new_cones = cones_to_worldframe(cones, gnss_offset, heading_offset, gnss_null, heading_null)

    gtmd_fixed = cones_to_worldframe_tf(all_gtmd_cones_dict['pos_with_color'],tmat,gnss_null, heading_null)
    # # Find closest timestamp in ego to world transformations
    new_gtmd_cones = cones_to_worldframe_tf(gtmd_cones, tmat,gnss_null, heading_null)
    new_cones = cones_to_worldframe_tf(cones, tmat,gnss_null, heading_null)

    # # car is at [0,0] in egomotion but at COG_OFFSET in local RTK frame
    # car_position = cones_to_worldframe(np.array([[0.0, 0.0, 0.0]]), gnss_offset, heading_offset, gnss_null, heading_null)[0, 1:]
    car_position = cones_to_worldframe_tf(np.array([[0.0, 0.0, 0.0]]), tmat,gnss_null, heading_null)[0, 1:]
    ## Start plotting

    # TODO build big dict so use pd transfer to Dataframe
    # 1. GTMD for global
    # 2. vehilce position
    # 3. GTMD visable
    # 4. matched cone
    # 5. Cone with no match
    # TODO add this to a function

    # add gtmd_fixed to list
    length = len(gtmd_fixed)
    x_gtmd_fixed = gtmd_fixed[:, 1].tolist()
    y_gtmd_fixed = gtmd_fixed[:, 2].tolist()
    group_gtmd_fixed = ['gtmd_fixed'] * length

    # car pos to list
    x_car = [car_position[0]]
    y_car = [car_position[1]]
    # print('x_car y car ',x_car, y_car)
    # x_car = [tmat[0,3]]
    # y_car = [tmat[1,3]]
    group_car = ['car']

    # visible GTMD cones to list
    length = len(new_gtmd_cones)
    x_new_gtmd = new_gtmd_cones[:, 1].tolist()
    y_new_gtmd = new_gtmd_cones[:, 2].tolist()
    group_new_gtmd = []
    # TODO transfer color without loop
    for cone in new_gtmd_cones:
        cone_color = kpi_utils.cone_color_to_plot_color(cone[0])
        group_new_gtmd.append('new_gtmd_' + cone_color)

    # new cones to list
    length = len(new_cones)
    x_new = new_cones[:, 1].tolist()
    y_new = new_cones[:, 2].tolist()
    group_new_cones = []
    for cone in new_cones:
        cone_color = kpi_utils.cone_color_to_plot_color(cone[0])
        group_new_cones.append('new_' + cone_color)

    # add unmatched gtmd cones
    unmatched_gtmd_cones = new_gtmd_cones[unmatched_gtmd_idx]
    length = len(unmatched_gtmd_cones)
    x_unmatched_gtmd = unmatched_gtmd_cones[:, 1].tolist()
    y_unmatched_gtmd = unmatched_gtmd_cones[:, 2].tolist()
    group_unmatched_gtmd = ['unmatched_gtmd'] * length

    # add multipled matched gtmd cones
    multiple_gtmd_cones = new_gtmd_cones[multiple_matched_gtmd_idx]
    length = len(multiple_gtmd_cones)
    x_multiple_gtmd = multiple_gtmd_cones[:, 1].tolist()
    y_multiple_gtmd = multiple_gtmd_cones[:, 2].tolist()
    group_multiple_gtmd = ['multiple_gtmd'] * length

    # TODO add line, hexagon and other shape
    # # draw red hexagon around multiple matched GTMDs
    # multiple_gtmds = new_gtmd_cones[multiple_matched_gtmd_idx]
    # length = len(multiple_gtmds)
    # x_multiple_gtmds = multiple_gtmds[:,1]
    # y_multiple_gtmds = multiple_gtmds[:,2]
    # shape_multi = ['p']*length  # pentagon
    # color_multi = ['red'] * length
    # TODO add size

    # build big dict
    frame_dict = {}
    frame_dict['x'] = x_gtmd_fixed + x_car + x_new_gtmd + x_new + x_unmatched_gtmd + x_multiple_gtmd
    frame_dict['y'] = y_gtmd_fixed + y_car + y_new_gtmd + y_new + y_unmatched_gtmd + y_multiple_gtmd
    frame_dict['cones group'] = group_gtmd_fixed + group_car + group_new_gtmd + group_new_cones + group_unmatched_gtmd + group_multiple_gtmd
    frame_data = pd.DataFrame(frame_dict)

    colors_manual = {
        "gtmd_fixed": 'grey',
        "car": 'green',
        'new_gtmd_blue': 'blue',
        'new_gtmd_gold': 'gold',
        'new_gtmd_darkorange': 'darkorange',
        'new_gtmd_moccasin': 'moccasin',
        'new_blue': 'blue',
        'new_gold': 'gold',
        'new_darkorange': 'darkorange',
        'new_moccasin': 'moccasin',
        'unmatched_gtmd': 'red',
        'multiple_gtmd': 'red',
    }

    shapes_manual = {
        "gtmd_fixed": '+',
        "car": 'o',
        'new_gtmd_blue': '+',
        'new_gtmd_gold': '+',
        'new_gtmd_darkorange': '+',
        'new_gtmd_moccasin': '+',
        'new_blue': '^',
        'new_gold': '^',
        'new_darkorange': '^',
        'new_moccasin': '^',
        'unmatched_gtmd': 'x',
        'multiple_gtmd': 's',
    }

    sizes_manual = {
        "gtmd_fixed": 3,
        "car": 5,
        'new_gtmd_blue': 3,
        'new_gtmd_gold': 3,
        'new_gtmd_darkorange': 3,
        'new_gtmd_moccasin': 3,
        'new_blue': 4,
        'new_gold': 4,
        'new_darkorange': 4,
        'new_moccasin': 4,
        'unmatched_gtmd': 4,
        'multiple_gtmd': 4,
    }

    current_velocity = "None"
    if velocity:
        best_ts = np.argmin(np.abs(np.array(list(t['time'] for t in velocity)) - timestamp))
        # Parentheses used here are a way to format a multiline string according to implied line continuation
        current_velocity = (f'x: {velocity[best_ts]["vel"][0]} y: {velocity[best_ts]["vel"][1]}'
                            f' theta: {velocity[best_ts]["vel"][2]}')

    if gtmd_fixed.size != 0:
        # set axis limits based on gtmd cones and some padding
        min_axis = np.min(gtmd_fixed[:, 1:3])
        max_axis = np.max(gtmd_fixed[:, 1:3])
    else:
        min_axis = -15
        max_axis = 15

    plot_text = f'''\
Total cones: {len(cones)} Total GT cones: {len(new_gtmd_cones)}
blue: {blue_cones_count:3d}:{blue_gtmd_count:3d} ({(blue_cones_count - blue_gtmd_count):3d}) yellow {yellow_cones_count:3d}:{yellow_gtmd_count:3d} ({(yellow_cones_count - yellow_gtmd_count):3d}) (cones:gt)(cones-gt)

True positive:  {correct_matches:3d}
False negative: {gtmd_no_match:3d}                   b:{unmatched_gtmd_counter.get(0, 0):3d}    y:{unmatched_gtmd_counter.get(1, 0):3d}
False positive: {(gtmd_multiple_match + unmatched_cones + cone_match_wrong_color):3d}
    GTMD with multiple matches: {gtmd_multiple_match:3d}    b:{multiple_matched_gtmd_counter.get(0, 0):3d}    y:{multiple_matched_gtmd_counter.get(1, 0):3d}
    Cone with no match:         {unmatched_cones:3d}    b:{no_matched_cones_counter.get(0, 0):3d}    y:{no_matched_cones_counter.get(1, 0):3d}
    Cone with wrong color:      {cone_match_wrong_color:3d}

Velocity: {current_velocity}
Frame: {current_frame}
Timestamp: {timestamp}
        '''

    plot = (
        ggplot(frame_data, aes(x='x', y='y', color='cones group', size='cones group', shape='cones group')) + geom_point() + scale_color_manual(values=colors_manual) +
        scale_size_manual(values=sizes_manual) + scale_shape_manual(values=shapes_manual) + annotate('text', x=min_axis - 20, y=max_axis - 10, label=plot_text) + xlim(min_axis - 40, max_axis) +
        ylim(min_axis, max_axis) + coord_fixed() + ggtitle(title) + xlab('Distance from start [m]') + ylab('Distance from start [m]') + theme_classic()  # theme_minimal  theme_classic
    )

    # either show plot or store it in a buffer for the video
    if create_video:
        #buf = io.BytesIO()
        #plot.save(filename=buf,height=7, width=12,format = 'png', dpi=200)
        # plot.save(filename=buf, format='png', dpi=200)
        # plot.draw()
        return plot

    plot.draw()
    return None
