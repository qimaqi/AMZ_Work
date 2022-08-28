#!/usr/bin/env python3
#
# AMZ Driverless Project
#
# Copyright (c) 2021 Authors:
#   - Qi MA <qimaqi@ethz.ch>
#
# All rights reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
#

import os
import shutil
import io
import re
import click
import cv2

import numpy as np
from PIL import Image
from . import constants as const
from plotnine import ggplot
# from plotnine.data import *


def init_video_dir() -> None:
    if os.path.isdir(const.VIDEO_FOLDER):
        if click.confirm('The video folder exists already do you want to delete it?'):
            click.echo("deleting folder ...")
            shutil.rmtree(const.VIDEO_FOLDER)
            os.makedirs(const.VIDEO_FOLDER)
    else:
        os.makedirs(const.VIDEO_FOLDER)


def save_buf(buf: io.BytesIO, current_frame: int) -> None:
    """
    Save the KPI video frame to certain file

    """

    buf.seek(0)
    img_from_buf = Image.open(buf)
    save_name = 'img_' + str(current_frame) + '.png'
    save_path = os.path.join(const.VIDEO_FOLDER, save_name)
    img_from_buf.save(save_path)


def save_ggplot(plot, current_frame: int) -> None:
    """
    Save the KPI video frame to certain file

    """
    save_name = str(current_frame).zfill(9) + '.png'
    save_path = os.path.join(const.VIDEO_FOLDER, save_name)
    plot.save(filename=save_path, width=12, height=12, dpi=200)


def video_maker(output='out.avi', frame_rate=1) -> None:
    save_folder = os.getcwd()

    images = [img for img in os.listdir(const.VIDEO_FOLDER) if img.endswith(".png")]
    # reformulate the imgs
    img_name_list = []
    for img_name in images:
        img_name_i = re.split('_', img_name)[-1]
        img_name_list.append(int(img_name_i[:-4]))
    idxs = np.argsort(img_name_list)
    frame = cv2.imread(os.path.join(const.VIDEO_FOLDER, images[0]))
    height, width, _ = frame.shape
    # fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    save_path = os.path.join(save_folder, output)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video = cv2.VideoWriter(save_path, fourcc, frame_rate, (width, height),isColor =True)

    for image_idx in idxs:
        video.write(cv2.imread(os.path.join(const.VIDEO_FOLDER, images[image_idx])))

    cv2.destroyAllWindows()
    video.release()
