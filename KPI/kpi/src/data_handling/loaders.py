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
# pylint: skip-file

from collections import OrderedDict
import os
import pickle

import numpy as np


def load_boundaries(folder_path):
    if not os.path.isdir(folder_path):
        raise RuntimeError(f"No folder called {folder_path} found.")
    files = os.listdir(folder_path)
    files = sorted(files)  # Sort chronologically
    if not files:
        raise RuntimeError(f"No cone arrays found in {folder_path}.")
    boundaries = OrderedDict()
    for file in files:
        file_path = os.path.join(folder_path, file)
        boundary = np.fromfile(file_path).reshape(-1, 5)
        timestamp = int(os.path.splitext(file)[0])
        boundaries[timestamp] = boundary
    return boundaries


def load_cone_arrays(folder_path):
    if not os.path.isdir(folder_path):
        raise RuntimeError(f"No folder called {folder_path} found.")
    files = os.listdir(folder_path)
    files = sorted(files)  # Sort chronologically
    if not files:
        raise RuntimeError(f"No cone arrays found in {folder_path}.")
    cone_arrays = OrderedDict()
    for file in files:
        file_path = os.path.join(folder_path, file)
        cone_array = np.fromfile(file_path).reshape(-1, 13)
        timestamp = int(os.path.splitext(file)[0])
        cone_arrays[timestamp] = cone_array
    return cone_arrays


def load_tf(folder_path):
    tf_path = os.path.join(folder_path, 'tf.pickle')
    with open(tf_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_StateDt(folder_path):
    with open(os.path.join(folder_path, 'StateDt.pickle'), 'rb') as f:
        data = pickle.load(f)
    return data


def load_gnss(folder_path):
    gnss_path = os.path.join(folder_path, 'gnss.npy')
    if not os.path.isfile(gnss_path):
        raise RuntimeError(f"File {gnss_path} doesn't exist.")
    gnss_array = np.fromfile(gnss_path).reshape(-1, 8)
    return gnss_array

    
def load_gnss_csv(folder_path):
    gnss_path = os.path.join(folder_path, 'gnss.csv')
    if not os.path.isfile(gnss_path):
        raise RuntimeError(f"File {gnss_path} doesn't exist.")
    gnss_array = np.genfromtxt(gnss_path).reshape(-1, 8)
    return gnss_array


def load_gtmd(folder_path):
    file_path = os.path.join(folder_path, 'gtmd.npy')
    if not os.path.isfile(file_path):
        raise RuntimeError(f"Can't retrieve GTMD data, no file called {file_path} found.")
    gtmd_array = np.fromfile(file_path).reshape(-1, 5)
    return gtmd_array


def load_velocities(folder_path):
    velocity_path = os.path.join(folder_path, 'velocity.npy')
    if not os.path.isfile(velocity_path):
        raise RuntimeError(f"File {velocity_path} doesn't exist.")
    velocities = np.fromfile(velocity_path).reshape(-1, 4)
    return velocities
