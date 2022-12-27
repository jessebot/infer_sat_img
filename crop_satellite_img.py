#!/usr/bin/env python3.11
"""
       Name: generate_512image.py
DESCRIPTION: Generate a small 512 bt 512 image from a giant tiff file
     AUTHOR: OverStory (slight modifications by Jesse Hitch)
"""
import argparse
import rasterio
from flask_app import utils


# this is a default that I've had downloaded already
SAT_TILE = './Sentinel2L2A_sen2cor_18TUR_20180812_clouds=5.3%_area=99%.tif'

# parse args
parsr = argparse.ArgumentParser()
parsr.add_argument("--sat_tile", "-s", type=str, default=SAT_TILE,
                   help="Path to satellite tile you'd like to process."
                        "Must end in .tif")
args = parsr.parse_args()


def crop_img(sat_tile=SAT_TILE):
    """
    Returns
    """
    if not sat_tile.endswith(".tif"):
        raise Exception("Only tif types allowed at this time.")

    # 512 by 512 is hardcoded right now, but we could change this in time
    crop = (5000, 5000, 512, 512)

    # second time
    image, meta = utils.tif_to_image(sat_tile, crop=crop)

    # meta needed to save the image and load all 10 bands
    with rasterio.open("cropped_img.tif", 'w', **meta) as dst:
        dst.write(image)

    return True


if __name__ == '__main__':
    crop_img(args.sat_tile)
