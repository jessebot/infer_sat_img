#!/usr/bin/env python3.11
"""
       Name: generate_512image.py
DESCRIPTION: Generate a small 512 bt 512 image from a giant tiff file
     AUTHOR: OverStory (slight modifications by Jesse Hitch)
"""
import rasterio
import utils


SAT_TILE = './Sentinel2L2A_sen2cor_18TUR_20180812_clouds=5.3%_area=99%.tif'


def main(sat_tile=SAT_TILE):
    """
    This is an example:
    # NOTE were only reading first rgb bands here
    # for model predictions, we skip the band argument to
    # load all 10 bands
    image, meta = utils.tif_to_image(sat_tile, crop=crop, bands=[3, 2, 1])

    # meta needed if we want to save the image
    with rasterio.open("test.tif", 'w', **meta) as dst:
        dst.write(image)
    """
    crop = (5000, 5000, 512, 512)

    # second time
    image, meta = utils.tif_to_image(sat_tile, crop=crop)

    # meta needed if we want to save the image
    with rasterio.open("test-full.tif", 'w', **meta) as dst:
        dst.write(image)

    return True


if __name__ == '__main__':
    main()
