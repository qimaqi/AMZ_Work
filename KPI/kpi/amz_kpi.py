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

import click

from kpi_dict import click_kpi_dict
from src.run_all.run_all import run_all

import cProfile


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.pass_context
def amz_kpi(ctx):  # pylint: disable=unused-argument
    '''
    AMZ KPI toolsuite
    '''


amz_kpi.add_command(run_all, name='run')
for kpi in click_kpi_dict.values():
    amz_kpi.add_command(kpi[0], name=kpi[1])

if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    cProfile.run("amz_kpi()")
