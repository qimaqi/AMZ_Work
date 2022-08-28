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

from typing import Optional, Union
import click
import yaml

from src.data_handling.rosbag_extractor import clean_up, extract_data

from .dummy_kpi import DummyKpi


@click.command()
@click.argument('rosbag', type=click.Path(exists=True))
@click.argument('gtmd_file', type=click.Path(exists=True))
@click.option('--config', '-c', type=click.Path(exists=True), help='Optional config file to store additional parameters.')
@click.option('--visualize', '-v', is_flag=True, help='Display the visualizations of this KPI.')
def click_run_dummy_kpi(rosbag: str, gtmd_file: str, config: Optional[str] = None, visualize: bool = False):
    '''
    This is a dummy KPI used as a reference implementation for the CLI.

    Just copy this file and start implementing your own KPI.
    Do not forget to add it to "kpi_dict.py".
    '''
    # Extract the required data from the rosbag
    print('-' * 10)
    extract_data(rosbag, gtmd_file)
    print('-' * 10)

    # Run the specified KPI
    run_dummy_kpi(config, visualize)

    # Delete the extracted data
    print('-' * 10)
    clean_up()


def run_dummy_kpi(config: Optional[Union[str, dict]] = None, visualize: bool = False):
    '''
    This function can also be called from another function since it does not have the click
    decorators.
    '''

    # Additional configuration has been passed in a YAML file
    if isinstance(config, str):
        config_file = config
        with open(config_file, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    # The parameters have already been read in the "run_all" method
    elif isinstance(config, dict):
        pass
    else:
        config = {}

    # Run the KPI
    dummy_kpi = DummyKpi(config)
    dummy_kpi.run()
    if visualize:
        dummy_kpi.visualize()
