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
# This file is handeling the perception KPIs
#

import pathlib
import signal as sig
import sys
from typing import Tuple, List
from types import FrameType
import re

import click

import numpy as np
# import ffmpeg
from scipy.interpolate import Akima1DInterpolator
from pyproj import Transformer
from tqdm import tqdm

from src.data_handling.rosbag_extractor import clean_up, extract_data
from src.data_handling.data_loader import DataLoader
from src.data_handling.image_analyst import init_video_dir, save_buf, video_maker, save_ggplot

from src.utils import kpi_utils
from .aggregate_statistics import aggregate_statistics
from .plot_track import plot_track_ggplot  # plot_track

from src.utils.preprocess_utils import build_sensor_fusion_2020_structure, build_fiter_2022_structure

from src.yolo_extension.detector import YoloDetector
from src.utils.filter_2022 import filtering_2022

import os
import glob


@click.command()
@click.argument('rosbag', type=click.Path(exists=True))
@click.argument('gtmd_file', type=click.Path(exists=True))
@click.argument('gnss_file', type=click.Path(exists=False))
@click.argument('pipeline_topic')
@click.option('--create_video', '-cv', is_flag=True, help='Toggle to generate a video. Default is off. '
              'Default name is "out.extension" and will overwrite existing file.')
@click.option('--track_plot_offset', '-tpo', default=-1, help='Frame offset when track plot and video should start. Default is the first frame')
@click.option('--track_plot_end', '-tpe', default=999999, help='End frame for track plot and video. Default is all the frames')
@click.option('--track_plot', '-tp', is_flag=True, help='Toggle whether the track should be plotted for each frame')
@click.option('--framerate', default=10, help='Framerate for the output video. Default is 10')
@click.option('--output', '-o', default='out.avi', help='Output name for plots or videos. Default is "out.extension" ')
@click.option('--title', default=None, type=str, help='Title for plots. Default is off')
@click.option('--plot_statistics', '-ps', is_flag=True, help='Toggle to plot statistics. Default is off and only textual output')
@click.option('--no_clean', '-nc', is_flag=True, help="Skip cleanup of extracted data")
@click.option('--no_extract', '-ne', is_flag=True, help="Skip extraction of data. This should be used if the data is already extracted")
@click.option('--matching_distances',
              '-md',
              type=str,
              default='0.5',
              help="Maximum matching distance between GTMD and detected cone. "
              "Default is 0.5 . Can be a comma separated list"
              " e.g. 0.2,0.5,1,2 ")
@click.option('--layout', '-l', type=click.Choice(['0', '1']), default='0', help='Layout type of plots')
@click.option('--buckets', '-b', type=str, default='(0,35)', help='List of ranges to consider e.g. (0,10),(10,20) '
              'The ranges are [inclusive, exclusive). Default'
              'is (0,35) meters')
@click.option('--lidar_no_color', '-lnc', is_flag=True, help='Do not consider color for matching, all cones are blue. Default is off and color is considered')
@click.option('--filter_2022','-f22', is_flag=True, help='Evaluate the filter based method 2022')
@click.option('--f_save_result','-f22s', is_flag=True, default = False, help='Evaluate the filter based method 2022')
@click.option('--conf_thres','-cft', default = 0.8, help = 'if using filter_2022 can also set confidence threshold')
@click.option('--iou_thres','-iot', default = 0.5, help = 'if using filter_2022 can also set iou threshold')
@click.option('--fssim','-fs',is_flag=True, help = 'Using the kpi tool to generate fssim statistic')


def click_run_perception(rosbag: pathlib.Path, gtmd_file: pathlib.Path, gnss_file:pathlib.Path, pipeline_topic: str, create_video: bool, framerate: int, output: str, title: str, track_plot_offset: int, track_plot_end: int,
                         track_plot: bool, plot_statistics: bool, no_clean: bool, no_extract: bool, matching_distances: str, layout: str, buckets: str, lidar_no_color: bool, filter_2022: bool,
                         f_save_result: bool, conf_thres:float, iou_thres: float, fssim:bool) -> None:
    """
    Entry point for the perception KPI function.
    This validates and extracts some arguments and
    then forwards them to run_perception()
    """
    # pylint: disable=too-many-arguments, too-many-locals

    # register function to handle sigint (ctrl + c usually)
    sig.signal(sig.SIGINT, sig_term_signal_handler)

    # extract buckets and put them in the bucket_list
    matches = re.findall(r'\((\d{1,2},\d{1,2})\)', buckets)
    bucket_list = []
    for match in matches:
        values = match.split(',')
        bucket_list.append((int(values[0]), int(values[1])))

    # extract comma separated list into array of floats
    matching_distances = list(float(d.strip()) for d in matching_distances.split(','))

    if framerate < 1:
        raise ValueError('Framerate should be larger than 0')

    if track_plot_offset >= track_plot_end:
        raise ValueError("Offset can't be larger or equal to end frame")

    # set the title for plots to the rosbag file name
    if title is None:
        title = 'Bag: ' + pathlib.Path(rosbag).stem + ' Topic: ' + pipeline_topic


    if not no_extract:
        extract_data(rosbag, gtmd_file,gnss_file, filter_2022)
        # filter 2022 kpi test
        if filter_2022 == True:
            # run 2020 sensor fusion extraction and preprocessing
            # base_path = build_sensor_fusion_2020_structure(rosbag, gtmd_file)
            # print('The data root', base_path)
            # run rosbag extraction
            # TODO
            file_path = pathlib.Path(__file__).parent.resolve()
            # rosbag_extract_path = os.path.join(file_path,'..','preprocessing','rosbag_extraction','main.py')
            # os.system("python3 " + rosbag_extract_path + " --extract_all -b " + base_path)
            # data_path = glob.glob(base_path+'/data/*/')
            # print('find data folder: ',[data_path])
            # rosbag_preprocess_path = os.path.join(file_path,'..','preprocessing','preprocessing','main.py')
            # os.system("python3 "+rosbag_preprocess_path+  " --match_data -d " +data_path[0])
            # build_fiter_2022_structure(data_path[0])


            yolov5 = YoloDetector(conf_thres, iou_thres)
            # generate labels 
            yolov5.bbox_inference()
            # using filter method to get new cone array

    run_perception(pipeline_topic, gnss_file, create_video, framerate, output, title, track_plot_offset, 
            track_plot, track_plot_end, plot_statistics, matching_distances, int(layout), bucket_list, lidar_no_color, filter_2022,f_save_result, conf_thres,iou_thres,fssim)

    if not no_clean:
        clean_up()


def sig_term_signal_handler(signal: int, frame: FrameType) -> None:
    # pylint: disable=unused-argument
    print('You pressed Ctrl+C!')
    sys.exit(0)


def get_gtmd_cones(data_loader: DataLoader,
                   timestamp: int,
                   bucket: Tuple[int, int],
                   gnss_interpolators: List[Akima1DInterpolator],
                   gnss_projection_trans: Transformer,
                   filter_fov: float = np.pi) -> List[List[float]]:
    """
    Loads, interpolates, and filters GTMD cones.

    :param data_loader: Dataloader from data_handling to load gtmd cones
    :param timestamp: Current timestamp in ns
    :param bucket: Bucket [min,max) which filters cones
    :param gnss_interpolators: Prebuild interpolator for GNSS data.
    Query get_gnss_interpolators() to build one
    :param gnss_projection_trans: Prebuild pyproj transformation object.
    Query get_transformer() to build one
    :param filter_fov: float if to filter the view.  >= 2pi and no filtering will happen.
    field of view.
    Useful if all GTMD should be returned
    :return: List of lists, where each inner list is a GTMD with [color, x, y]
    and within the bucket distance and fov
    """

    gtmd_cones_loaded = data_loader.get_gtmd()  # [color, x, y, ??, ??] in lat/long

    matching_gnss = kpi_utils.interp_gnss_provided_interpolators(timestamp, gnss_interpolators)
    # print('matching_gnss',matching_gnss)

    longitude_avg, latitude_avg = matching_gnss[1:3]
    heading_avg = matching_gnss[7]
    # [color, x, y] in egomotion frame. 0,0 is the car position
    gtmd_cones = kpi_utils.gtmd_to_ego_frame(gtmd_cones_loaded, longitude_avg, latitude_avg, heading_avg, gnss_projection_trans)

    # 175° = 3.054326rad for MRH LiDAR. 165° for FW LiDAR
    # Mono camera has 79.8° but unsure about total FOV of all 3
    if 0 < filter_fov <= 2 * np.pi:
        gtmd_cones = kpi_utils.filter_cones_by_fov(gtmd_cones, filter_fov)

    # Filter cones so that they are in the bin. Bins are inclusive/exclusive
    squared = np.square(gtmd_cones[:, 1:])
    distances = np.sqrt(squared.sum(1))  # 1 is the axis, this sums horizontally
    return gtmd_cones[(bucket[0] <= distances) & (distances < bucket[1])]

def get_gtmd_cones_tf(data_loader: DataLoader,
                   timestamp: int,
                   bucket: Tuple[int, int],
                   longitude_avg_start,
                   latitude_avg_start,
                   heading_start,
                   tmat,
                   filter_fov: float = np.pi) -> List[List[float]]:
    """
    Loads, interpolates, and filters GTMD cones.

    :param data_loader: Dataloader from data_handling to load gtmd cones
    :param timestamp: Current timestamp in ns
    :param bucket: Bucket [min,max) which filters cones
    :param gnss_interpolators: Prebuild interpolator for GNSS data.
    Query get_gnss_interpolators() to build one
    :param gnss_projection_trans: Prebuild pyproj transformation object.
    Query get_transformer() to build one
    :param filter_fov: float if to filter the view.  >= 2pi and no filtering will happen.
    field of view.
    Useful if all GTMD should be returned
    :return: List of lists, where each inner list is a GTMD with [color, x, y]
    and within the bucket distance and fov
    """

    # TODO
    # 1. all cones in rtk coordinate
    # 2. all cone from rtk frame to ego frame: rtk gtmd to rtk orgin. rtk orgin go to vehicle frame by tf
    # 
    gtmd_cones_loaded = data_loader.get_gtmd()  # [color, x, y, ??, ??] in lat/long
    
    # gtmd to world frame
    gtmd_transformed = kpi_utils.gtmd_to_ego_frame(gtmd_cones_loaded, longitude_avg_start, latitude_avg_start, heading_start, kpi_utils.get_transformer())
    # change gtmd to egomotion frame
    gtmd_cones = kpi_utils.gtmd_to_ego_frame2(gtmd_transformed, tmat)

    # matching_gnss = kpi_utils.interp_gnss_provided_interpolators(timestamp, gnss_interpolators)
    # print('matching_gnss',matching_gnss)

    # longitude_avg, latitude_avg = matching_gnss[1:3]
    # heading_avg = matching_gnss[7]
    # [color, x, y] in egomotion frame. 0,0 is the car position
    # gtmd_cones = kpi_utils.gtmd_to_ego_frame(gtmd_cones_loaded, longitude_avg, latitude_avg, heading_avg, gnss_projection_trans)

    # 175° = 3.054326rad for MRH LiDAR. 165° for FW LiDAR
    # Mono camera has 79.8° but unsure about total FOV of all 3
    if 0 < filter_fov <= 2 * np.pi:
        gtmd_cones = kpi_utils.filter_cones_by_fov(gtmd_cones, filter_fov)

    # Filter cones so that they are in the bin. Bins are inclusive/exclusive
    squared = np.square(gtmd_cones[:, 1:])
    distances = np.sqrt(squared.sum(1))  # 1 is the axis, this sums horizontally
    return gtmd_cones[(bucket[0] <= distances) & (distances < bucket[1])]


def plot_and_get_frame(data_loader: DataLoader, timestamp: int, gnss_interpolators: Akima1DInterpolator, gnss_projection_trans: Transformer, frame_data: dict, bucket: Tuple[int, int],
                       matching_distance: float, create_video: bool, title: str, current_frame: int, velocity: List[dict], longitude_avg_start,latitude_avg_start, gnss_null: Tuple[float, float], heading_null: float, tmat, layout: int):
    """
    The function calling the plotting given all the parameters. It is mainly a wrapper to make
    the code a bit more readable:
    :param data_loader: The dataloader used
    :param timestamp: the current timestamp
    :param gnss_interpolators: An interpolator for the GNSS data
    :param gnss_projection_trans: Prebuild pyproj transformation object.
    Query get_transformer() to build one
    :param frame_data: A dict with timestamps containing dicts about cones, gtmd_cones,
    matched cones and other cone matching data
    :param bucket: A tupel of the range bucket, e.g. (0,25) - 0m-25m
    :param matching_distance: the used matching distance
    :param create_video: If a video should be created
    :param title: the title of the plot
    :param current_frame: the current frame of the video or sequence of plots
    :param velocity: a list of dicts. Each dict has the keys: time:float, vel: List[float],
    acc: List[float], vel_voc: List[float]
    :param gnss_null: gnss_null
    :param heading_null: the original heading
    :param layout: what layout to use
    """

    #pylint: disable=too-many-arguments
    # extract all the GTMD cones and GNSS data for the current position
    all_gtmd_cones = get_gtmd_cones(data_loader, timestamp, (0, 9999), gnss_interpolators, gnss_projection_trans, 2 * np.pi)
    # all_gtmd_cones = get_gtmd_cones_tf(data_loader, timestamp, (0, 9999), longitude_avg_start,latitude_avg_start,heading_null,tmat, 2 * np.pi)
    gnss_current = kpi_utils.interp_gnss_provided_interpolators(timestamp, gnss_interpolators)
    gnssxy = gnss_projection_trans.transform(gnss_current[1], gnss_current[2])

    # return value is all_gtmd_cones
    # np.array(gnssxy), gnss_current[7], gnss_null, heading_null, layout)
    # return value is None if create_video = false and shouldn't be used

    frame = plot_track_ggplot(frame_data[timestamp], all_gtmd_cones, timestamp, bucket, matching_distance, create_video, title, current_frame, velocity, np.array(gnssxy), gnss_current[7], gnss_null,
                              heading_null,tmat,layout)
    return frame


def plot_and_get_frame_tf(data_loader: DataLoader, timestamp, tmat, longitude_avg_start,latitude_avg_start,heading_start, frame_data: dict, bucket: Tuple[int, int],
                       matching_distance: float, create_video: bool, title: str, current_frame: int, velocity: List[dict], gnss_null: Tuple[float, float], heading_null: float, layout: int):
    """
    The function calling the plotting given all the parameters. It is mainly a wrapper to make
    the code a bit more readable:
    :param data_loader: The dataloader used
    :param timestamp: the current timestamp
    :param gnss_interpolators: An interpolator for the GNSS data
    :param gnss_projection_trans: Prebuild pyproj transformation object.
    Query get_transformer() to build one
    :param frame_data: A dict with timestamps containing dicts about cones, gtmd_cones,
    matched cones and other cone matching data
    :param bucket: A tupel of the range bucket, e.g. (0,25) - 0m-25m
    :param matching_distance: the used matching distance
    :param create_video: If a video should be created
    :param title: the title of the plot
    :param current_frame: the current frame of the video or sequence of plots
    :param velocity: a list of dicts. Each dict has the keys: time:float, vel: List[float],
    acc: List[float], vel_voc: List[float]
    :param gnss_null: gnss_null
    :param heading_null: the original heading
    :param layout: what layout to use
    """
    #pylint: disable=too-many-arguments
    # extract all the GTMD cones and GNSS data for the current position
    gtmd_cones_loaded = data_loader.get_gtmd()
    # gtmd to tf world frame
    gtmd_transformed = kpi_utils.gtmd_to_ego_frame(gtmd_cones_loaded, longitude_avg_start, latitude_avg_start, heading_start, kpi_utils.get_transformer())
    # all_gtmd_cones = get_gtmd_cones(data_loader, timestamp, (0, 9999), gnss_interpolators, gnss_projection_trans, 2 * np.pi)
    
    current_x = tmat[0,3]
    current_y = tmat[1,3]
    # gnss_current = kpi_utils.interp_gnss_provided_interpolators(timestamp, gnss_interpolators)
    # gnssxy = gnss_projection_trans.transform(gnss_current[1], gnss_current[2])

    # return value is all_gtmd_cones
    # np.array(gnssxy), gnss_current[7], gnss_null, heading_null, layout)
    # return value is None if create_video = false and shouldn't be used

    frame = plot_track_ggplot(frame_data[timestamp], gtmd_transformed, timestamp, bucket, matching_distance, create_video, title, current_frame, velocity, np.array(gnssxy), gnss_current[7], gnss_null,
                              heading_null, layout)
    return frame


def run_perception(pipeline_topic: str, gnss_file:pathlib.Path, create_video: bool, framerate: int, output_name: str, title: str, track_plot_offset: int, track_plot: bool, track_plot_end: int, plot_statistics: bool,
                   matching_distances: List[float], layout: int, bins: List[Tuple[int, int]], lidar_no_color: bool,filter_2022: bool, f_save_result:bool, conf_thres: float,iou_thres:float, fssim:bool) -> None:
    """
    Main function of perception KPI tool. Extracts the chosen topic
    and computes statistics on it for each bucket and each matching distance.
    Default output is textual aggregated statistics. Other options are plots or a video.
    Not all parameters are checked for validity/consistency as this function is
    generally only called via click_run_perception which forwards the arguments.
    Defaults are also stored in this function and not here so that no duplicate
    code has to be maintained.

    :param pipeline_topic: Name of the pipeline topic to compute statistics on.
        Usually something like /perception/lidar/cone_array
    :param create_video: Flag to produce a video
    :param framerate: Framerate of the produced video
    :param output_name: Output name of the video
    :param title: Title for the plots
    :param track_plot_offset: Offset frame when the plot/video should start
    :param track_plot: Flag to plot the track for each frame
    :param track_plot_end: End frame for the plot/video
    :param plot_statistics: Flag to create statistics plots
    :param matching_distances: Array of matching distances that are considered
    :param lidar_no_color: Flag to consider no color
    """
    # pylint: disable=too-many-locals, too-many-arguments

    # load data
    data_loader = DataLoader()
    if gnss_file.endswith('.csv'):
        gnss = data_loader.align_gnss()
    else:
        gnss = data_loader.get_topic("/pilatus_can/GNSS")

    
    try:
        velocity = data_loader.get_topic("/pilatus_can/velocity_estimation")
    except FileNotFoundError:
        velocity = None

        # Get transforms
    transform_dict = data_loader.get_topic("/tf")
    egomotion_to_world = np.array(transform_dict['egomotion_to_world']).astype(np.float128)
    
    # driving_intervals = kpi_utils.get_driving_intervals(velocity, 0)

    if filter_2022 == True:
        filter_instance = filtering_2022(conf_thres = conf_thres, save_result = f_save_result, lidar_type='compensated_pc')
        filter_instance.get_cone_array()
        #cProfile.run('filter_instance.get_cone_array()')
        pipeline_topic = '/perception/sensor_fusion_2022/cone_array'
    cones = data_loader.get_topic(pipeline_topic)

    # precompute the interpolators to speedup computation for the GTMD cones
    gnss_interpolators = kpi_utils.get_gnss_interpolators(gnss)
    # precompute transformation to speedup computation in gnss_to_egomotion
    gnss_projection_trans = kpi_utils.get_transformer()

    # hardcoded settings
    cone_confidence = 0

    # Average the 50 first GNSS points to get a better estimate.
    # This position will be the origin in our worldframe
    longitude_avg_start, latitude_avg_start = np.average(gnss[0:100, 1:3], axis=0)
    # heading_avg_start = np.average(gnss[:100, 7])
    heading_start = np.average(gnss[0:100, 7])

    longitude_avg, latitude_avg = np.average(gnss[0:50, 1:3], axis=0)
    gnss_null = gnss_projection_trans.transform(longitude_avg, latitude_avg)
    heading_null = np.average(gnss[0:50, 7])
    

    # sys.exit()

    if create_video:
        # launch python ffmpeg as subprocess:
        # https://kkroening.github.io/ffmpeg-python/#ffmpeg.run_async
        # Add from Qi
        init_video_dir()
        # ffmpeg_subprocess = (
        #     ffmpeg.input('pipe:', pix_fmt='rgb24', framerate=framerate, loglevel="quiet").output(
        #         output_name + '.webm', pix_fmt='yuv420p', lossless=1,
        #         loop=0)  # yuv420p for mp4 and webm, bgra for animatedwebp
        #     # don't use gif, that's a shitty format and you need palletegen to make it work
        #     .overwrite_output().run_async(pipe_stdin=True))

    # get a progress bar only if we show plots or create a video otherwise it is annoying
    if create_video or track_plot:
        dict_iterator = tqdm(cones.items(), total=len(cones.keys()), desc="timestamp", position=0, leave=True)
        bucket_iterator = tqdm(bins, total=len(bins), desc='Buckets')
    else:
        dict_iterator = cones.items()
        bucket_iterator = bins
    for bucket in bucket_iterator:
        # dict contains or each matching_distance a frame_data
        # dict which contains data for each individual frame
        data_by_matching_distance = {}
        for matching_distance in matching_distances:
            frame_data = {}  # called frame data as it refers to one "camera frame"
            errors_x = []
            errors_y = []
            theta_list = []
            dist_list = []
            range_error = []
            bearing_error = []
            FP_range = []
            FP_bearing = []
            FN_range = []
            FN_bearing = []
            TP_range = []
            TP_bearing = []
            # process each cone array
            current_frame = 0
            # if it is possible to start after moving

            for timestamp, cone_array in dict_iterator:

                # driving only
                if True: #kpi_utils.is_time_in_driving_interval(timestamp, driving_intervals):
  
                    tmat = kpi_utils.get_tmat_at_timestamp(egomotion_to_world, timestamp)
                    if pipeline_topic == "/perception/lidar/cone_array":
                        fov = np.pi
                    elif pipeline_topic == "/perception/mono_camera/cone_array":
                        fov = np.pi*135/180
                    elif pipeline_topic == "/perception/cone_array":
                        fov = np.pi*79.8/180
                    
                    # gtmd_cones = get_gtmd_cones_tf(data_loader, timestamp, bucket, longitude_avg_start,latitude_avg_start,heading_start,tmat, filter_fov=fov)
                    gtmd_cones = get_gtmd_cones(data_loader, timestamp, bucket, gnss_interpolators, gnss_projection_trans,filter_fov=fov)
                    
                    # print('gtmd_cones_tf',gtmd_cones_tf)
                    # print('gtmd_cones',gtmd_cones)
                    # Turn all gtmd cones and cone array cones to color 0 (bule)
                    if lidar_no_color:
                        gtmd_cones[:, 0] = 0
                        cone_array[:, 1] = 1
                        cone_array[:, 2] = 0
                        cone_array[:, 3] = 0
                        cone_array[:, 4] = 0
                    cone_dict = kpi_utils.unpack_cones(cone_array, cone_confidence=cone_confidence, bucket=bucket)
                    frame_data[timestamp] = kpi_utils.get_confusion_data(gtmd_cones, cone_dict, matching_distance)
                    # error_i_x, error_i_y = kpi_utils.get_offset(gtmd_cones,cone_dict, matching_distance)

                    if fssim:
                        error_i_x =  frame_data[timestamp]['offset_x']
                        error_i_y =  frame_data[timestamp]['offset_y']
                        theta_i = frame_data[timestamp]['theta']
                        distance_r = frame_data[timestamp]['distance_r']
                        range_error_i = frame_data[timestamp]['range_list']
                        bearing_error_i = frame_data[timestamp]['bearing_list']
                        if len(error_i_y) != 0 and len(error_i_x) != 0:
                            errors_x.extend(error_i_x)
                            errors_y.extend(error_i_y)
                            theta_list.extend(theta_i)
                            dist_list.extend(distance_r)
                            range_error.extend(range_error_i)
                            bearing_error.extend(bearing_error_i)
                        
                        FP_range_i = frame_data[timestamp]['FP_distribution']['range']
                        FP_bearing_i = frame_data[timestamp]['FP_distribution']['bearing']
                        FN_range_i = frame_data[timestamp]['FN_distribution']['range']
                        FN_bearing_i = frame_data[timestamp]['FN_distribution']['bearing']
                        TP_range_i = frame_data[timestamp]['TP_distribution']['range']
                        TP_bearing_i = frame_data[timestamp]['TP_distribution']['bearing']

                        if len(FP_range_i) != 0:
                            FP_range.extend(FP_range_i)
                            FP_bearing.extend(FP_bearing_i)
                        
                        if len(FN_range_i) != 0:
                            FN_range.extend(FN_range_i)
                            FN_bearing.extend(FN_bearing_i)

                        if len(TP_range_i) != 0:
                            TP_range.extend(TP_range_i)
                            TP_bearing.extend(TP_bearing_i)
                        
                    if (create_video or track_plot) and track_plot_offset < current_frame < track_plot_end:
                        # Plot track or create video
                        # frame = plot_and_get_frame(data_loader, timestamp, gnss_interpolators,
                        #                            gnss_projection_trans, frame_data, bucket,
                        #                            matching_distance, create_video, title,
                        #                            current_frame, velocity, gnss_null, heading_null,
                        #                            layout)
                        frame = plot_and_get_frame(data_loader, timestamp, gnss_interpolators, gnss_projection_trans, frame_data, bucket, matching_distance, create_video, title, current_frame, velocity,
                                                longitude_avg_start,latitude_avg_start,gnss_null, heading_null, tmat, layout)
                        # frame = plot_and_get_frame_tf(data_loader, timestamp, tmat, longitude_avg_start,latitude_avg_start,heading_start, frame_data, bucket, matching_distance, create_video, title, current_frame, velocity,
                        #                         gnss_null, heading_null, layout)

                        if create_video and frame:
                            # save_buf(frame, current_frame)
                            save_ggplot(frame, current_frame)
                        #     # send frame as bytes to ffmpeg
                        #     ffmpeg_subprocess.stdin.write(frame.getvalue())

                    current_frame += 1

            # 
            errors_x_np = np.array(errors_x)
            errors_y_np = np.array(errors_y)
            theta_list_np = np.array(theta_list)
            dist_list_np = np.array(dist_list)
            range_error_np = np.array(range_error)
            bearing_error_np = np.array(bearing_error)
            # print(errors_x_np.flatten(), errors_y_np.flatten())
            # debug drawing
            data_by_matching_distance[matching_distance] = frame_data
        aggregate_statistics(data_by_matching_distance, bucket, plot_statistics)
        import matplotlib.pyplot as plt

        if fssim:
            plt.scatter(errors_x_np.flatten(), errors_y_np.flatten())
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig('/home/qimaqi/Desktop/distribution.png')
            plt.clf()

            X_1 = np.stack((errors_x_np.flatten(),errors_y_np.flatten()), axis = 0)
            print('conv distribution in xy ',np.cov(X_1))

            plt.scatter(errors_x_np.flatten()/dist_list_np.flatten(), errors_y_np.flatten()/dist_list_np.flatten())
            plt.xlabel('x_norm')
            plt.ylabel('y_norm')
            plt.savefig('/home/qimaqi/Desktop/normalized_distribution.png')
            plt.clf()

            plt.scatter(range_error_np.flatten(), bearing_error_np.flatten())
            plt.xlabel('range')
            plt.ylabel('bearing')
            plt.savefig('/home/qimaqi/Desktop/range_bearing_distribution.png')
            plt.clf()

            X_2 = np.stack((errors_x_np.flatten()/dist_list_np.flatten(),errors_y_np.flatten()/dist_list_np.flatten()), axis = 0)
            print('conv distribution',np.mean(X_2,axis=1),np.cov(X_2))
            print('X_2',np.shape(X_2))

            X_3 = np.stack((range_error_np.flatten(),bearing_error_np.flatten()), axis = 0)
            print('conv distribution mean',np.mean(X_3,axis=1))
            print('cov distribution', np.cov(X_3))

            plt.scatter(np.array(FP_range).flatten(), np.array(FP_bearing).flatten())
            plt.xlabel('range')
            plt.ylabel('bearing')
            plt.savefig('/home/qimaqi/Desktop/range_bearing_FP_distribution.png')
            plt.clf()

            plt.scatter(np.array(FN_range).flatten(), np.array(FN_bearing).flatten())
            plt.xlabel('range')
            plt.ylabel('bearing')
            plt.savefig('/home/qimaqi/Desktop/range_bearing_FN_distribution.png')
            plt.clf()

            plt.scatter(np.array(TP_range).flatten(), np.array(TP_bearing).flatten())
            plt.xlabel('range')
            plt.ylabel('bearing')
            plt.savefig('/home/qimaqi/Desktop/range_bearing_TP_distribution.png')
            plt.clf()

            X_4 = np.stack((np.array(FP_range).flatten(),np.array(FP_bearing).flatten()), axis = 0)
            print('False positive cov distribution mean',np.mean(X_4,axis=1))
            print('False positive cov distribution', np.cov(X_4))
            
            X_5 = np.stack((np.array(FN_range).flatten(),np.array(FN_bearing).flatten()), axis = 0)
            print('False negative conv distribution mean',np.mean(X_5,axis=1))
            print('False negative cov distribution', np.cov(X_5))

            X_6 = np.stack((np.array(TP_range).flatten(),np.array(TP_bearing).flatten()), axis = 0)
            print('True positive conv distribution mean',np.mean(X_6,axis=1))
            print('True positive  distribution', np.cov(X_6))

            # save TP files
            TP_range_path = '/home/qimaqi/Desktop/TP_range.npy'
            TP_bearing_path = '/home/qimaqi/Desktop/TP_bearing.npy'
            TP_range_np = np.array(TP_range).flatten()
            TP_bearing_np = np.array(TP_bearing).flatten()
            TP_range_np.tofile(TP_range_path)
            TP_bearing_np.tofile(TP_bearing_path)

            # save FP
            FP_range_path = '/home/qimaqi/Desktop/FP_range.npy'
            FP_bearing_path = '/home/qimaqi/Desktop/FP_bearing.npy'
            FP_range_np = np.array(FP_range).flatten()
            FP_bearing_np = np.array(FP_bearing).flatten()
            FP_range_np.tofile(FP_range_path)
            FP_bearing_np.tofile(FP_bearing_path)

            # save FN
            FN_range_path = '/home/qimaqi/Desktop/FN_range.npy'
            FN_bearing_path = '/home/qimaqi/Desktop/FN_bearing.npy'
            FN_range_np = np.array(FN_range).flatten()
            FN_bearing_np = np.array(FN_bearing).flatten()
            FN_range_np.tofile(FN_range_path)
            FN_bearing_np.tofile(FN_bearing_path)

        # plt.scatter(errors_x_np.flatten()*np.theta_list_np, errors_y_np.flatten())
        # plt.xlabel('x_norm_r')
        # plt.ylabel('y_norm_r')
        # plt.savefig('/home/qimaqi/Desktop/r_distribution.png')
        # plt.clf()



    if create_video:
        # close stream and wait for ffmpeg to finish processing
        # ffmpeg_subprocess.stdin.close()
        # ffmpeg_subprocess.wait()
        video_maker(output=output_name, frame_rate=framerate)