#!/usr/bin/env python3.11
"""
       Name: crop_satellite_img.py
DESCRIPTION: Creates a cropped image from a larger satelitte image
     AUTHOR: OverStory (modifications by Jesse Hitch)
"""
import argparse
import rasterio
import logging as log
from sys import stderr


# this is a default that I've had downloaded already
SAT_TILE = './Sentinel2L2A_sen2cor_18TUR_20180812_clouds=5.3%_area=99%.tif'

# set logging before we import the utils, so it uses the same basic config
FORMAT = '%(asctime)s [%(levelname)s] %(funcName)s: %(message)s'
log.basicConfig(stream=stderr, format=FORMAT, level=log.INFO)
log.info("Infer Sat Image Logging Config Loaded.")

from flask_app import utils  # noqa: E402 - needs to load after logging


def crop_img(sat_tile=SAT_TILE, width=512, height=512, x_pos=0, y_pos=0):
    """
    Creates a cropped image from a larger satellite image
    Saves image called cropped_img_{width}x{height}at{x_pos}x{y_pos}y.tif
                     ┌───────────────────────┐
                     │ ┌─── width ──────┐    │ 
                     │ │ ☁  🌳 ☁  ☁     H    │ 
                     │ │  ☁  ☁      ☁   e    │ 
    satellite img ➡  │ │         🌳     i    │↕
                     │ │🌳 Cropped img  g    │y-axis for location of crop
                     │ │                h    │
                     │ │  ☁  ☁   🌳     t    │
                     │ └────────────────┘    │
                     │                       │
                     └───────────────────────┘
                     ↔ x-axis for location of crop
    Optional args:
        sat_tile -  which is a filename of a satelite tif
        width    -  width of the crop you want to create
        height   -  height of the crop you want to create
        x_pos    -  x-axis coridinate of crop starting point
        y_pos    -  y-axis cordinate of crop starting point
    """
    if not sat_tile.endswith(".tif"):
        raise Exception("Only tif types allowed at this time.")

    # crop = (5000, 5000, width, height)
    crop = (x_pos, y_pos, width, height)

    log.info(f"{crop}")

    # second time
    image, meta = utils.tif_to_image(sat_tile, crop=crop)

    cropped_tif = f'cropped_img_{width}x{height}at{x_pos}x{y_pos}y.tif'
    # meta needed to save the image and load all 10 bands
    with rasterio.open(cropped_tif, 'w', **meta) as dst:
        dst.write(image)

    return True


if __name__ == '__main__':
    # parse args
    pars = argparse.ArgumentParser(prog='create_satellite_img.py',
                                   description="Create 🌾🌾🌾🌾 from 🛰️  🖼️ ",
                                   epilog='Mostly written by Overstory, '
                                          'but also Jesse a little bit.')

    pars.add_argument("--crop_width", "-w", type=int, default='512',
                      help="Width for the crop. Defaults to 512. "
                           "(must be increment of 64)")

    pars.add_argument("--crop_height", "-H", type=int, default='512',
                      help="Height for the crop. Defaults to 512. "
                           "(must be increment of 64)")

    pars.add_argument("--x_position", "-x", type=int, default='0',
                      help="Position on the x-axis to place crop. Default: 0")

    pars.add_argument("--y_position", "-y", type=int, default='0',
                      help="Position on the y-axis to place crop. Default: 0")

    pars.add_argument("--sat_tile", "-s", type=str, default=SAT_TILE,
                      help="Path to satellite tile you'd like to process."
                           "Must end in .tif")
    args = pars.parse_args()

    crop_img(args.sat_tile,
             args.crop_width, args.crop_height,
             args.x_position, args.y_position)
