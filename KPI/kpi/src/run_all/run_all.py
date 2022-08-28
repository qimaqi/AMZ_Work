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

import os
from pathlib import Path
import click
import yaml

from kpi_dict import kpi_dict
from src.data_handling.rosbag_extractor import clean_up, extract_data

try:
    DEFAULT_CONFIG_FILE = Path(os.environ['AMZ_ROOT']) / 'tools/kpi/amz_kpi_config.yaml'
except KeyError as exception:
    DEFAULT_CONFIG_FILE = None


@click.command()
@click.option('--config',
              '-c',
              'config_file',
              required=DEFAULT_CONFIG_FILE is None,
              default=str(DEFAULT_CONFIG_FILE) if DEFAULT_CONFIG_FILE is not None else None,
              show_default=True,
              type=click.Path(exists=True),
              help='Path to an alternate config file.')
def run_all(config_file: str):
    '''
    Run multiple KPIs specified in the config file.
    '''

    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Check for unknown KPIs and alert the user
    for kpi in config['kpi']:
        if kpi['type'] not in kpi_dict:
            click.echo(click.style(f'Found unknown KPI of type: {kpi["type"]} --> name: {kpi["name"]}', fg='red'))

    # Iterate over all given rosbags
    for data in config['data']:
        print('-' * 10)
        extract_data(data['rosbag'], data['gtmd_file'])
        print('-' * 10)

        # Compute all KPIs for the current rosbag
        for kpi in config['kpi']:
            print(f'Run {kpi["name"]}')
            kpi_dict[kpi['type']](kpi['config'])
            print('-' * 10)

        # Delete tmp folder
        print('-' * 10)
        clean_up()
