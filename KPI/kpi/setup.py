#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
# pylint: skip-file

from setuptools import find_packages
from setuptools import setup


def read_long_description():
    with open('README.md', 'r', encoding="utf8") as file:
        long_description = file.read()
    return long_description


setup(
    name='amz-kpi',
    author='Niclas Vödisch',
    version='1.0',
    author_email='vniclas@ethz.ch',
    description='AMZ KPI toolsuite',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://bitbucket.org/amzracing/autonomous_2021',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'click', 'colorama', 'gnupg', 'numpy', 'matplotlib', 'pycryptodome', 'pycryptodomex', 'pyproj==2.6', 'pyyaml', 'rospkg', 'scikit-learn', 'scipy', 'tqdm', 'scikits.bootstrap', 'ffmpeg-python'
    ],
    scripts=['amz_kpi.py'],
    entry_points='''
        [console_scripts]
        amz_kpi=amz_kpi:amz_kpi
    ''',
)
