#!/usr/bin/env python3
#
# AMZ Driverless Project
#
# Copyright (c) 2021 Authors:
#   - Niclas Vödisch <vniclas@ethz.ch>
#
# All rights reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
#

from collections import defaultdict

# ToDo: New KPIs need to be imported and added to both dictionaries
from src.dummy_kpi.run_dummy_kpi import run_dummy_kpi, click_run_dummy_kpi
from src.slam_accuracy.run_slam_accuracy import run_slam_accuracy, click_run_slam_accuracy
from src.be_flicker.run_be_flicker import run_be_flicker, click_run_be_flicker
from src.be_accuracy.run_be_accuracy import run_be_accuracy, click_run_be_accuracy
from src.perception.run_perception import run_perception, click_run_perception

# This dictionary contains the mapping from a KPI type to the corresponding runner function
kpi_dict = defaultdict(lambda: lambda x: x)
kpi_dict['dummy_kpi'] = run_dummy_kpi
kpi_dict['slam_accuracy'] = run_slam_accuracy
kpi_dict['be_flicker'] = run_be_flicker
kpi_dict['be_accuracy'] = run_be_accuracy
kpi_dict['perception'] = run_perception

# This dictionary contains the mapping from a KPI type to the corresponding click runner function
# 1st value: runner function called when the KPI is started from the terminal
# 2nd value: how the KPI should be started from the terminal, e.g.,
#            "amz_kpi dummy_kpi <optional args>"
click_kpi_dict = {}  #dict()
click_kpi_dict['dummy_kpi'] = (click_run_dummy_kpi, 'dummy_kpi')
click_kpi_dict['slam_accuracy'] = (click_run_slam_accuracy, 'slam_accuracy')
click_kpi_dict['be_flicker'] = (click_run_be_flicker, 'be_flicker')
click_kpi_dict['be_accuracy'] = (click_run_be_accuracy, 'be_accuracy')
click_kpi_dict['perception'] = (click_run_perception, 'perception')
