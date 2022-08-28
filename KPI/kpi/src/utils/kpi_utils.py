#!/usr/bin/env python3
#
# AMZ Driverless Project
#
# Copyright (c) 2021 Authors:
#   - Stefan Weber <stefwebe@ethz.ch>
#   - Thierry Backes <tbackes@ethz.ch>
#   - Michele Graziano <michelgr@ethz.ch>
#
# All rights reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
#

from typing import Optional, List, Tuple, Union
from collections import Counter
import math

import numpy as np

from scipy import interpolate
from scipy.spatial.transform import Rotation

from sklearn.neighbors import NearestNeighbors

from pyproj import Transformer
import matplotlib.pyplot as plt

RTK_COG_OFFSET = 1.166  # meter



def get_gnss_interpolators(gnss_data) -> List[interpolate.Akima1DInterpolator]:
    """
    Creates Akima1DInterpolators for each field of the GNSS data except
    the timestamp. Building this interpolator is expensive and should
    only be done once.
    """
    interpolators = []
    for idx in range(1, gnss_data.shape[1]):
        f = interpolate.Akima1DInterpolator(gnss_data[:, 0], gnss_data[:, idx])
        interpolators.append(f)

    return interpolators


def interp_gnss_provided_interpolators(timestamp: float, interpolators):
    """
    Interpolate GNSS at the given timestamp with the provided interpolators
    Provide interpolators that are fit on the full GNSS dataset because
    the construction of them is an expensive operation and they stay invariant
    for a given rosbag. Interpolators can be created by get_gnss_interpolators()
    in kpi_utils
    """

    # add +1 as there is no interpolator for timestamp,
    # but the output array should have the timestamp field
    ref_gnss_data = np.zeros(len(interpolators) + 1)
    ref_gnss_data[0] = timestamp
    for column_idx in range(1, len(interpolators) + 1):  # start at 1 as we don't interpolate the timestamp
        ref_gnss_data[column_idx] = interpolators[column_idx - 1](timestamp)  # interpolators are 0 indexed
    return ref_gnss_data


def interp_gnss(ref_timestamp, gnss_data):
    """`
    Given two timestamps and the gnss data, interpolate
    the gnss data at the ref_timestamp with cubic polynomials.
    """
    ref_gnss_data = np.zeros(gnss_data[0].shape)
    ref_gnss_data[0] = ref_timestamp
    for column_idx in range(1, len(ref_gnss_data)):
        # Fits the function *exactly* through the gnss points, so not robust
        # to outliers?
        f = interpolate.Akima1DInterpolator(gnss_data[:, 0], gnss_data[:, column_idx])
        ref_gnss_data[column_idx] = f(ref_timestamp)
    return ref_gnss_data


def get_transformer() -> Transformer:
    """
    Returns a pyproj transformer: https://pyproj4.github.io/pyproj/dev/api/transformer.html
    that fits to Switzerland/Central Europe. Object should be stored rather than recreated
    every time it is required
    """
    return Transformer.from_crs(
        "epsg:4326",
        "+proj=utm +zone=32 +ellps=WGS84",
        always_xy=True,
    )

def gtmd_to_ego_frame2(gtmd_transformed, tmat):
    """
    Given the gtmd in amz world frame we need to do two thing
    """
    # project lat/long data using the provided transformer
    # print('gtmd_transformed',gtmd_transformed) [color, x, y]
    cone_pos_world = gtmd_transformed[:,1:3]
    # homography
    # add z channel

    cols = np.shape(cone_pos_world)[0]
    cone_pos_world = np.append(cone_pos_world, np.zeros((cols,1)),axis=1)
    # cone_pos_world = np.append(cone_pos_world, np.ones((cols,1)),axis=1)

    # change world to egomotion
    rot_mat = np.linalg.inv(tmat[:3,:3])
    # print('rot_mat',rot_mat)
    # print('tmat[:,:3]',np.shape(tmat[:3,3]))
    trans = np.matmul(-rot_mat, tmat[:3,3])
    # print('trans',trans)

    cone_pos_ego = np.matmul(rot_mat, cone_pos_world.T).T
    cone_pos_ego += trans
    cone_pos_ego_2d = cone_pos_ego[:,:2]
    cone_pos_ego_2d = np.column_stack((gtmd_transformed[:,0], cone_pos_ego_2d))   
    gtmd_2d = cone_pos_ego_2d

    # # print('cone_pos_ego_2d',cone_pos_ego_2d)

    # gnss_x, gnss_y = transformer.transform(gnss_long, gnss_lat)
    # gnss_2d = np.array([gnss_x, gnss_y])
    # x_values, y_values = transformer.transform(gtmd_cones[:, 2], gtmd_cones[:, 1])  # [x, y]
    # gtmd_2d = np.column_stack([x_values, y_values])

    # # Transform cones with gnss_2d and gnss heading
    # gtmd_2d -= gnss_2d
    # phi = (gnss_heading + 90) / 180.0 * np.pi
    # rot_mat = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    # gtmd_2d = np.matmul(rot_mat, gtmd_2d.T).T

    # # Now cones are in the RTK frame. Finally, transform into ego frame
    # tf_rtk_to_vehicle = [RTK_COG_OFFSET, 0]
    # gtmd_2d += tf_rtk_to_vehicle
    # gtmd_2d = np.column_stack((gtmd_cones[:, 0], gtmd_2d))  # [color, x, y]
    return gtmd_2d


def gtmd_to_ego_frame(gtmd_cones, gnss_long, gnss_lat, gnss_heading, transformer):
    """
    Given the raw gtmd cones and a single gnss data point (longitude and latitude), project both to
    2D and transform the gtmd cones to the RTK frame.
    Output format: (Color, x, y)

    transformer is a pyproj transformer: https://pyproj4.github.io/pyproj/dev/api/transformer.html
    that has to be provided. This is faster than repeatedly applying a Proj() to each object as
    this builds a transformer object internally for each call. Transformer can be generated by
    get_transformer() function in kpi_utils
    """
    # project lat/long data using the provided transformer
    gnss_x, gnss_y = transformer.transform(gnss_long, gnss_lat)
    gnss_2d = np.array([gnss_x, gnss_y])
    x_values, y_values = transformer.transform(gtmd_cones[:, 2], gtmd_cones[:, 1])  # [x, y]
    gtmd_2d = np.column_stack([x_values, y_values])

    # Transform cones with gnss_2d and gnss heading
    gtmd_2d -= gnss_2d
    phi = (gnss_heading + 90) / 180.0 * np.pi
    rot_mat = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    gtmd_2d = np.matmul(rot_mat, gtmd_2d.T).T

    # Now cones are in the RTK frame. Finally, transform into ego frame
    tf_rtk_to_vehicle = [RTK_COG_OFFSET, 0]
    gtmd_2d += tf_rtk_to_vehicle
    gtmd_2d = np.column_stack((gtmd_cones[:, 0], gtmd_2d))  # [color, x, y]
    return gtmd_2d


def get_driving_intervals(velocity_data: np.ndarray, minimum_speed_threshold: float) -> List[Tuple[int, int]]:
    """
    Given a minimum absolute velocity threshold, return all the [starst_time, end_time]
    tuples where the car is driving
    """
    start_time = []
    end_time = []
    is_driving = False
    for velocity in velocity_data:
        vel_i = velocity['vel']
        timestamp_i = velocity['time']
        speed = np.linalg.norm(vel_i[1:3])

        if speed > minimum_speed_threshold and not is_driving:
            start_time.append(timestamp_i)  # get the timestamp
            is_driving = True
        if speed < minimum_speed_threshold and is_driving:
            end_time.append(timestamp_i)
            is_driving = False
    # velocity logs end while the car is still running use the last time stamp as end time
    if is_driving:
        # end_time.append(velocity_data[len(velocity_data) - 1, 0])
        end_time.append(velocity_data[len(velocity_data) - 1]['time'])
    if len(start_time) > 0 and len(start_time) == len(end_time):
        driving_intervals = list(zip(start_time, end_time))
    else:
        raise RuntimeWarning("No driving intervals found for the given threshold")
    return driving_intervals


def is_time_in_driving_interval(time: float, intervals: List[Tuple[float, float]]) -> bool:
    """
    Checks if the time is in one of the driving intervals. Assumes that for (a,b) we have
    a <= b
    """
    return any(map(lambda i: i[0] <= time <= i[1], intervals))


def bin_cones_by_distance(cone_array: np.ndarray, bins: Optional[Union[np.ndarray, List[float], Tuple[float, ...]]] = None, bin_length: Optional[float] = None) -> np.ndarray:
    if len(cone_array.shape) != 2 or cone_array.shape[1] != 13:
        raise ValueError(f'Illegal cone array of shape: {cone_array.shape}')
    if bins is not None and len(bins) < 2:
        raise ValueError(f'Bins are not valid intervals: {bins}')
    if bin_length is not None and bin_length <= 0:
        raise ValueError(f'Bin length less than or equal to zero: {bin_length}')
    if bins is None and bin_length is None:
        raise ValueError('Specify either bins or bin_length.')

    distance = np.linalg.norm(cone_array[:, 6:8], axis=1)  # L2 distance from (x, y) position
    if bins is None:

        # We add bin_length to the stop to ensure that the max value will actually be in a bin.
        bins = np.arange(0, distance.max() + bin_length, bin_length)
    bin_indices = np.digitize(distance, bins)
    # We assign -1 to cones that are not in any of the requested bins
    bin_indices[distance < bins[0]] = -1
    bin_indices[distance > bins[-1]] = -1
    return bin_indices


def match_two_cone_arrays(gt_array, lookup_array, upper_bound, k: int = 1, algorithm: str = 'brute'):
    """
    function takes a ground truth cone array and an array to lookup cones.
    the cone array has to be composed in the following way: [color, x,y,z]
    x,y,z in [m]
    for each cone in the lookup array it returnes the indices of the closest
    neighbour and the euclidean distance to the matching cone.

    If a cone has no neighbor, distance is Inf and index is len(gt_array)

    upper_bound doesn't match cones that are further away than this value

    Algorithm can be chosen from ['auto', 'ball_tree', 'kd_tree', 'brute']
    default is 'brute' as most cone arrays are small and the cost of building
    a kd tree is too much
    """

    nbrs = NearestNeighbors(n_neighbors=k, algorithm=algorithm).fit(gt_array[:, 1:3])
    dis, ind = nbrs.kneighbors(lookup_array[:, 1:3])

    # filter out cones that are larger than the upper bound. Set values
    # the same way that the scipy implementation set them to avoid
    # breaking the API
    out_of_range = (dis > upper_bound).flatten()
    dis[out_of_range] = np.Inf
    ind[out_of_range] = len(gt_array)

    return dis.flatten(), ind.flatten()


def unpack_gtmd_cones(gtmd_cones: np.array):
    """
    Unpacks the transformed gtmd_cones which are in [color, x, y] into a dict.

    pos_with_color holds an array [color, x, y, z] and the other keys are
    index arrays that point to cones with the spcecific color. They are:
    yellow_idx, blue_idx, orange_small_idx, orange_big_idx

    Colors:
    0: blue
    1: yellow
    2: orange small
    3: orange big
    """

    cones_pos = []
    cones_yellow_idx = []
    cones_blue_idx = []
    cones_orange_small_idx = []
    cones_orange_big_idx = []

    for idx, cone in enumerate(gtmd_cones):
        cones_pos.append(cone)
        if cone[0] == 0:
            cones_blue_idx.append(idx)
        elif cone[0] == 1:
            cones_yellow_idx.append(idx)
        elif cone[0] == 2:
            cones_orange_small_idx.append(idx)
        elif cone[0] == 3:
            cones_orange_big_idx.append(idx)

    cones_pos = np.array(cones_pos)
    cones_yellow_idx = np.array(cones_yellow_idx)
    cones_blue_idx = np.array(cones_blue_idx)
    cones_orange_small_idx = np.array(cones_orange_small_idx)
    cones_orange_big_idx = np.array(cones_orange_big_idx)

    return {'pos_with_color': cones_pos, 'yellow_idx': cones_yellow_idx, 'blue_idx': cones_blue_idx, 'orange_small_idx': cones_orange_small_idx, 'orange_big_idx': cones_orange_big_idx}


def unpack_cones(cone_array, cone_confidence: float, bucket=None):
    """
    Unpacks the cones into a dict.
    pos_with_color holds an array [color, x, y, z] and the other keys are
    index arrays that point to cones with the spcecific color. They are:
    yellow_idx, blue_idx, orange_small_idx, orange_big_idx

    Input cone_array has the following fields:
    id_cone, prob_type{blue, yellow, orange, orange_big}, prob_cone,
    position{x, y, z}, position_covariance{x_x, y_y, x_y}, is_observed

    Color is selected by using argmax(blue, yellow, orange, orange_big) confidences

    cone_confidence value filters cones by their prob_cone that are below the provided value

    If bucket is given, only store cones that are in the range of the bucket. Bucket contains
    bucket[0] minimum range (inclusive) and bucket[1] maximum range (exclusive)
    """

    cones_pos = []
    cones_yellow_idx = []
    cones_blue_idx = []
    cones_orange_small_idx = []
    cones_orange_big_idx = []

    for idx, cone in enumerate(cone_array):
        if bucket is not None:
            # skip cone if it's not in the current bucket. Inclusive/Exclusive range
            if not bucket[0] <= math.sqrt(cone[6]**2 + cone[7]**2) < bucket[1]:
                continue

        # discard cones below provided confidence value
        if cone[5] < cone_confidence:
            continue

        color = np.argmax(cone[1:5])
        if color == 0:
            cones_blue_idx.append(idx)
        elif color == 1:
            cones_yellow_idx.append(idx)
        elif color == 2:
            cones_orange_small_idx.append(idx)
        elif color == 3:
            cones_orange_big_idx.append(idx)

        cones_pos.append(np.concatenate([[color], cone[6:9]]))

    cones_pos = np.array(cones_pos)
    cones_yellow_idx = np.array(cones_yellow_idx)
    cones_blue_idx = np.array(cones_blue_idx)
    cones_orange_small_idx = np.array(cones_orange_small_idx)
    cones_orange_big_idx = np.array(cones_orange_big_idx)

    return {'pos_with_color': cones_pos, 'yellow_idx': cones_yellow_idx, 'blue_idx': cones_blue_idx, 'orange_small_idx': cones_orange_small_idx, 'orange_big_idx': cones_orange_big_idx}


def cone_color_to_plot_color(cone_color: int) -> str:
    """
    Helper to plot cones. Converts cone IDs to matplotlib
    colors. Raises error if color is not in [0,3]
    """

    if cone_color == 0:
        color = 'blue'
    elif cone_color == 1:
        color = 'gold'
    elif cone_color == 2:
        color = 'darkorange'
    elif cone_color == 3:
        color = 'moccasin'
    else:
        raise ValueError("Cone has color id: " + str(cone_color) + " which is not mapped")
    return color


# def get_offset(gtmd_cones, cone_dict,matching_distance):
#     '''
#     - 'cone_dict': original input cone dict
#     - 'gtmd_cones_dict': unpacked GTMD cones dictionary
#     '''
#     cones = cone_dict['pos_with_color']

#     if len(cones) == 0 and len(gtmd_cones) == 0:
#         return 0,0


#     _, matched_cones_idx = match_two_cone_arrays(gtmd_cones, cones, upper_bound=matching_distance, k=1)
#     for cone_idx, gt_idx in enumerate(matched_cones_idx):
#         if gt_idx == len(gtmd_cones):
#             # skip if cone has no matching GTMD
#             continue
#         cone_color = cones[cone_idx,0]
#         pos_cone_x = cones[cone_idx,1]
#         pos_cone_y = cones[cone_idx,2]
#         gtmd_color = gtmd_cones[gt_idx,0]
#         pos_gtmd_x = gtmd_cones[gt_idx,1]
#         pos_gtmd_y = gtmd_cones[gt_idx,2]
 
#         if cone_color == gtmd_color:
#             # the match is correct save the offset
#             offset_x.append(pos_cone_x - pos_gtmd_x)
#             offset_y.append(pos_cone_y - pos_gtmd_y)
#         ### debug drawing

#     return offset_x, offset_y



def get_confusion_data(gtmd_cones, cone_dict, matching_distance: float):
    # pylint: disable=too-many-branches,too-many-locals
    """
    Matches GTMD cones and provided cones and computes basic statistics. Output is a dict
    which contains statistics (scalars) or arrays of indices that correspond to certain
    statistics.

    Inputs are:
    - gtmd_cones: Array of [color, x, y]
    - cone_dict: Result of unpack_cones() function
    - matching_distance: Maximum distance between a GTMD cone and a detected one to be matched

    Ouput dict:
    - 'cone_dict': original input cone dict
    - 'gtmd_cones_dict': unpacked GTMD cones dictionary
    - 'correct_matches': Number of correct matches. This means that GTMD and cone are NN below
                         threshold and the colors are identical
    - 'gtmd_no_match': Number of GTMD cones without a match
    - 'unmatched_gtmd_idx': Indices of GTMD cones with no match
    - 'multiple_matched_gtmd_idx': Indices of GTMD cones with multiple matches
    - 'unmatched_cones_idx': Indices of cones that don't have a match
    - 'unmatched_gtmd_counter': Counter for GTMD cones without a match
    - 'multiple_matched_gtmd_counter': Counter for GTMD colors with multiple matches
    - 'no_matched_cones_counter': Counter for cones colors without a match
    - 'gtmd_multiple_match': Number of GTMD cones that have multiple matches
    - 'unmatched_cones': Number of cones that don't have a match
    - 'cone_match_wrong_color': Number of cones that matched but have the wrong color
    - 'matched_cones_idx': Return array from match_two_cone_arrays()
                           Index represents the cone, value
                           represents the GTMD index. If value = len(gtmd_cones)
                           then there is no match
    """

    # store cones separately as this will be used a lot
    cones = cone_dict['pos_with_color']

    ## Handle 3 edgecases first: No cones and no GTMD cones, No cones, No GTMD cones
    ## No GTMDs/Cones can happen if we set a too high threshold or chose a wrong binning distance

    FP_distribution = {}
    FN_distribution = {}
    TP_distribution = {}
    for key in ['range','bearing']:
        FP_distribution[key] = []
        FN_distribution[key] = []
        TP_distribution[key] = []

    if len(cones) == 0 and len(gtmd_cones) == 0:
        # Terminate early as we are blind in an empty world
        return {
            'cone_dict': cone_dict,
            'gtmd_cones_dict': unpack_gtmd_cones(gtmd_cones),
            'correct_matches': 0,
            'gtmd_no_match': 0,
            'unmatched_gtmd_idx': [],
            'multiple_matched_gtmd_idx': [],
            'unmatched_cones_idx': [],
            'unmatched_gtmd_counter': Counter(),
            'multiple_matched_gtmd_counter': Counter(),
            'no_matched_cones_counter': Counter(),
            'gtmd_multiple_match': 0,
            'unmatched_cones': 0,
            'cone_match_wrong_color': 0,
            'matched_cones_idx': [],
            'offset_x':[],
            'offset_y':[],
            'theta':[],
            'distance_r':[],
            'range_list':[],
            'bearing_list':[],
            'FP_distribution':FP_distribution,
            'FN_distribution':FN_distribution,
            'TP_distribution':TP_distribution
        }

    if len(cones) == 0:
        # Terminate early as there are no cones
        return {
            'cone_dict': cone_dict,
            'gtmd_cones_dict': unpack_gtmd_cones(gtmd_cones),
            'correct_matches': 0,
            'gtmd_no_match': len(gtmd_cones),
            'unmatched_gtmd_idx': list(range(len(gtmd_cones))),
            'multiple_matched_gtmd_idx': [],
            'unmatched_cones_idx': [],
            'unmatched_gtmd_counter': Counter([gtmd[0] for gtmd in gtmd_cones]),
            'multiple_matched_gtmd_counter': Counter(),
            'no_matched_cones_counter': Counter(),
            'gtmd_multiple_match': 0,
            'unmatched_cones': 0,
            'cone_match_wrong_color': 0,
            'matched_cones_idx': [],
            'offset_x':[],
            'offset_y':[],
            'theta':[],
            'distance_r':[],
            'range_list':[],
            'bearing_list':[],
            'FP_distribution':FP_distribution,
            'FN_distribution':FN_distribution,
            'TP_distribution':TP_distribution
        }

    if len(gtmd_cones) == 0:
        # Terminate early as there are no GTMD cones
        return {
            'cone_dict': cone_dict,
            'gtmd_cones_dict': unpack_gtmd_cones(gtmd_cones),
            'correct_matches': 0,
            'gtmd_no_match': 0,
            'unmatched_gtmd_idx': [],
            'multiple_matched_gtmd_idx': [],
            'unmatched_cones_idx': list(range(len(cones))),
            'unmatched_gtmd_counter': Counter(),
            'multiple_matched_gtmd_counter': Counter(),
            'no_matched_cones_counter': Counter([cone[0] for cone in cones]),
            'gtmd_multiple_match': 0,
            'unmatched_cones': len(cones),
            'cone_match_wrong_color': 0,
            'matched_cones_idx': [],
            'offset_x':[],
            'offset_y':[],
            'theta':[],
            'distance_r':[],
            'range_list':[],
            'bearing_list':[],
            'FP_distribution':FP_distribution,
            'FN_distribution':FN_distribution,
            'TP_distribution':TP_distribution
        }

    # Match GTMD and detected cones. Each position of the indices array represents one cone, the
    # value is the index of the GTMD cone. Note that indices of value len(gtmd_cones)
    # are cones that have no matching partner (as there is no GTMD with index len(gtmd_cones))
    _, matched_cones_idx = match_two_cone_arrays(gtmd_cones, cones, upper_bound=matching_distance, k=1)
    ## Counters and arrays for statistics
    unmatched_cones = 0
    unmatched_cones_idx = []

    gtmd_no_match = 0
    gtmd_multiple_match = 0
    multiple_matched_gtmd_idx = []
    unmatched_gtmd_idx = []

    matched_gtmd = [0] * len(gtmd_cones)  # array counts how often a GTMD cone got matches

    cone_match_wrong_color = 0
    correct_matches = 0

    # Arrays hold true and predicted colors for matched cones for confusion matrix
    color_true = []
    color_pred = []

    offset_x = []
    offset_y = []
    theta = []
    distance_r = []
    range_list = []
    bearing_list = []


    # Store the predicted and true colors to build a confusion matrix
    # Only store colors of matched cones
    # Calculate the no matched cone
    for cone_idx, cone in enumerate(cones):
        if cone_idx not in matched_cones_idx:
            # unmatched cone idx
            pos_cone_x = cones[cone_idx,1]
            pos_cone_y = cones[cone_idx,2]
            FP_distribution['range'].append(math.sqrt(pos_cone_x*pos_cone_x + pos_cone_y*pos_cone_y))
            FP_distribution['bearing'].append(math.atan2(pos_cone_x,pos_cone_y))

    for cone_idx, gt_idx in enumerate(matched_cones_idx):
        # TODO no matching better expression
        if gt_idx == len(gtmd_cones):
            # skip if cone has no matching GTMD
            continue
        color_pred.append(cones[cone_idx, 0])
        color_true.append(gtmd_cones[gt_idx, 0])

    # Compute unmatched and matched cones
    for cone_idx, gt_idx in enumerate(matched_cones_idx):
        if gt_idx != len(gtmd_cones):
            matched_gtmd[gt_idx] += 1  # GTMD cone has a match

        if gt_idx == len(gtmd_cones):
            # Detect cones that have no matching GTMD
            unmatched_cones += 1
            unmatched_cones_idx.append(cone_idx)
        elif cones[cone_idx, 0] == gtmd_cones[gt_idx, 0]:
            # cone matches with gtmd and has correct color
            correct_matches += 1
            pos_cone_x = cones[cone_idx,1]
            pos_cone_y = cones[cone_idx,2]
            pos_gtmd_x = gtmd_cones[gt_idx,1]
            pos_gtmd_y = gtmd_cones[gt_idx,2]
            TP_distribution['range'].append(math.sqrt(pos_gtmd_x*pos_gtmd_x + pos_gtmd_y*pos_gtmd_y))
            TP_distribution['bearing'].append(math.atan2(pos_gtmd_x,pos_gtmd_y))
            offset_x.append(pos_cone_x - pos_gtmd_x)
            offset_y.append(pos_cone_y - pos_gtmd_y)
            theta.append(math.atan2(pos_gtmd_x,pos_gtmd_y))
            distance_r.append(math.sqrt(pos_gtmd_x*pos_gtmd_x + pos_gtmd_y*pos_gtmd_y))
            range_list.append(math.sqrt(pos_cone_x*pos_cone_x + pos_cone_y*pos_cone_y) - math.sqrt(pos_gtmd_x*pos_gtmd_x + pos_gtmd_y*pos_gtmd_y))
            bearing_list.append(math.atan2(pos_cone_x,pos_cone_y) - math.atan2(pos_gtmd_x,pos_gtmd_y))

        else:
            # cone matches with gtmd but has wrong color
            cone_match_wrong_color += 1

    # Compute details about GTMD cones
    for gt_idx, total_matches in enumerate(matched_gtmd):
        if total_matches == 0:
            # GTMD has no match
            gtmd_no_match += 1
            unmatched_gtmd_idx.append(gt_idx)
            pos_gtmd_x = gtmd_cones[gt_idx,1]
            pos_gtmd_y = gtmd_cones[gt_idx,2]
            FN_distribution['range'].append(math.sqrt(pos_gtmd_x*pos_gtmd_x + pos_gtmd_y*pos_gtmd_y))
            FN_distribution['bearing'].append(math.atan2(pos_gtmd_x,pos_gtmd_y))
        elif total_matches > 1:
            # GTMD has more than one match
            gtmd_multiple_match += 1
            multiple_matched_gtmd_idx.append(gt_idx)

    # Count how often each color occurs
    unmatched_gtmd_counter = Counter([gtmd_cones[idx][0] for idx in unmatched_gtmd_idx])
    multiple_matched_gtmd_counter = Counter([gtmd_cones[idx][0] for idx in multiple_matched_gtmd_idx])
    no_matched_cones_counter = Counter([cones[idx][0] for idx in unmatched_cones_idx])

    return {
        'cone_dict': cone_dict,
        'gtmd_cones_dict': unpack_gtmd_cones(gtmd_cones),
        'correct_matches': correct_matches,
        'gtmd_no_match': gtmd_no_match,
        'unmatched_gtmd_idx': unmatched_gtmd_idx,
        'multiple_matched_gtmd_idx': multiple_matched_gtmd_idx,
        'unmatched_cones_idx': unmatched_cones_idx,
        'unmatched_gtmd_counter': unmatched_gtmd_counter,
        'multiple_matched_gtmd_counter': multiple_matched_gtmd_counter,
        'no_matched_cones_counter': no_matched_cones_counter,
        'gtmd_multiple_match': gtmd_multiple_match,
        'unmatched_cones': unmatched_cones,
        'cone_match_wrong_color': cone_match_wrong_color,
        'matched_cones_idx': matched_cones_idx,
        'offset_x':offset_x,
        'offset_y':offset_y,
        'theta': theta,
        'distance_r':distance_r,
        'range_list':range_list,
        'bearing_list':bearing_list,
        'FP_distribution':FP_distribution,
        'FN_distribution':FN_distribution,
        'TP_distribution':TP_distribution
    }


def filter_cones_by_fov(cones: np.ndarray, angle_rad: float, offset_m: float = 0) -> np.ndarray:
    """
    Given cones in egomotion frame and an angle it will remove all the cones which are not in the
    symmetric triangle spanned by the angle
    offset used for the distance between egomotion and the frame of the cones (x facing forward)
    angle should be < 0 and <2 pi, 0 means seeing nothing, 2pi means everything is visible
    Output format: cone array (n,13)
    """
    if not 0 <= angle_rad <= 2 * np.pi:
        raise ValueError("angle should be between 0 and 2pi")
    opp_angle = np.pi - angle_rad / 2
    cone_array_angle = np.abs(np.arctan2(cones[:, 2], -(cones[:, 1] - offset_m)))
    cone_array_new = cones[cone_array_angle > opp_angle]
    return cone_array_new


def get_tmat_at_timestamp(transform_data: np.ndarray, timestamp_ns: int) -> np.ndarray:
    """
    Given a numpy array of tranform data for a specific parent-child pair,
    return the transformation matrix at a given timestamp.
    """

    # print('transform_data[:, 0] ',transform_data[:, 0] )
    # print('timestamp_ns',timestamp_ns)
    # 1622997016045964032
    # 1622997029332927225

    tmat = np.zeros((4, 4))
    # index = np.where(transform_data[:, 0] == timestamp_ns)
    # find the closest one
    time_diff = np.abs(transform_data[:, 0]-timestamp_ns)
    nearest_index = np.argmin(time_diff)
    index = nearest_index
    # print('time difference between cone timestamps and nearest tf',time_diff[nearest_index])
    if index is None:
        raise RuntimeError("No transform found: check that transform and boundary timestamps match")

    quaternion = transform_data[index, 4:8].reshape((1, 4))
    # print('quaternion',quaternion)
    tmat[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    tmat[:3, 3] = transform_data[index, 1:4]
    # print('tmat',tmat)
    return tmat
