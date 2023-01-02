#!/usr/bin/env python3.11
"""
       Name: app_logger.py
DESCRIPTION: very quickly setup logger to be used in flask app and utils
"""
import logging as log
from sys import stderr


# set logging before we import the utils, so it uses the same basic config
FORMAT = '%(asctime)s [%(levelname)s] %(funcName)s: %(message)s'
log.basicConfig(stream=stderr, format=FORMAT, level=log.INFO)
log.info("Infer Sat Image Logging Config Loaded.")
