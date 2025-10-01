import os
import time
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from multiprocessing import Pool


def _evo_parallel(image):
    # open sentinel2 tile
    file = image.name
    tmp = outpath / 'evoland/experiment_resolution_redone'

    with rasterio.open(image) as src:
        data = src.read()
        reordered = data[[2, 1, 0, 3], :, :]

        profile = src.profile
        profile.update({"compress": "zstd",
                        "zstd_level": 13,
                        "interleave": "band"})

        with rasterio.open(tmp / file, "w", **profile) as dst:
            dst.write(reordered)
    return


def _sr4rs_parallel(image):
    # open sentinel2 tile
    file = image.name
    tmp = outpath / 'sr4rs' / 'tmp'

    with rasterio.open(image) as src:
        data = src.read()

        profile = src.profile
        profile.update({"compress": "zstd",
                        "zstd_level": 13,
                        "interleave": "band"})

        with rasterio.open(tmp / file, "w", **profile) as dst:
            dst.write(data)

    return



if __name__ == '__main__':

    # define strtification file for fps
    outpath = Path('/data/USERS/shollend/sentinel2/sr_inference')


    def evoland():
        # shuffle bands to rgbnir and compress
        model = 'evoland/experiment_resolution'
        images = [image for image in tqdm(Path(outpath / model).glob('*.tif'))]

        with Pool(processes=64) as pool:
            results = list(tqdm(pool.imap_unordered(_evo_parallel, images), total=len(images), desc="Processing"))


    def sr4rs():
        # compress and overwrite
        model = 'sr4rs'

        images = [image for image in tqdm(Path(outpath / model).glob('*.tif'))]

        with Pool(processes=32) as pool:
            results = list(tqdm(pool.imap_unordered(_sr4rs_parallel, images), total=len(images), desc="Processing"))


    sr4rs()