#!/usr/bin/env python

import sys

from setuptools import setup

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, (
    "Safety Gym is designed to work with Python 3.6 and greater. "
    + "Please install it before proceeding."
)

setup(
    name="safety_gym",
    packages=["safety_gym"],
    install_requires=["joblib~=0.14.0", "xmltodict~=0.13.0",],
)
