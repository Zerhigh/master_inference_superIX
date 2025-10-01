import rasterio
from rasterio.transform import Affine
from tqdm import tqdm
import os
from pathlib import Path
import torch
from multiprocessing import Pool
from functools import partial

from superIX.evoland.utils import run_evoland
from superIX.sr4rs.utils import run_sr4rs
from superIX.swin2_mose.utils import run_swin2_mose

def parallel(table, config):
    rows = [row for row in table['lr_s2_path']]

    with Pool(processes=os.cpu_count()) as pool:
        func = partial(_parallel, config=config)
        results = list(tqdm(pool.imap_unordered(func, rows), total=len(rows), desc="Processing"))
    return


def _parallel(file, config):
    # open sentinel2 tile
    filename = Path(file).name
    try:
        with rasterio.open(file) as src:
            img = src.read()
            config['src'] = src
            config['filename'] = filename
            logic(img, config)
    except:
        print(f'Error with image: {file}, skipping it')

    return


def sequential(table, config):
    for i, file in tqdm(enumerate(table['lr_s2_path'])):
        # open sentinel2 tile
        filename = Path(file).name
        try:
            with rasterio.open(file) as src:
                img = src.read()
                config['src'] = src
                config['filename'] = filename
                logic(img, config)
        except:
            print(f'Error with image: {file}, skipping it')
            continue

    return


def logic(img, config):
    # unpack config
    model = config['model']
    model_name = config['model_name']
    src = config['src']
    outpath = config['outpath']
    filename = config['filename']

    if model_name == 'evoland':
        if Path(outpath / model_name / 'evoland_resolution' / filename).exists():
            return
        result = run_evoland(model, lr=img)
        superX = result['sr']
        # output is returned as float32, but values are uint16
        # convert to float32 by division of 10_000
        superX = superX / 10_000

        # resample from 5m to 2.5m
        superX_resampled = torch.nn.functional.interpolate(
            torch.from_numpy(superX).unsqueeze(0),
            scale_factor=2,
            mode='bilinear',
            antialias=True
        ).squeeze().numpy()

        save_img(img=superX, src=src, save_path=outpath / model_name / 'evoland_resolution' / filename, res=5)
        save_img(img=superX_resampled, src=src, save_path=outpath / model_name / 'experiment_resolution' / filename,
                 res=2.5)

    elif model_name == 'sr4rs':
        if Path(outpath / model_name / filename).exists():
            return
        result = run_sr4rs(model, lr=img)
        save_img(img=result, src=src, save_path=outpath / model_name / filename, res=2.5)

    elif model_name == 'swin2_mose':
        if Path(outpath / model_name / 'swin2_mose_resolution' / filename).exists():
            return
        # returns r, g,b, nir at 5m
        results = run_swin2_mose(model, lr=img)
        results = results / 10_000
        # convert to flaot32

        # resample from 5m to 2.5m
        results_resampled = torch.nn.functional.interpolate(
            torch.from_numpy(results).unsqueeze(0),
            scale_factor=2,
            mode='bilinear',
            antialias=True
        ).squeeze().numpy()

        save_img(img=results, src=src, save_path=outpath / model_name / 'swin2_mose_resolution' / filename, res=5)
        save_img(img=results_resampled, src=src, save_path=outpath / model_name / 'swin2_mose_experiment' / filename,
                 res=2.5)
    return


def save_img(img, src, save_path, res):
    new_transform = Affine(res, src.transform.b, src.transform.c,
                           src.transform.d, -res, src.transform.f)

    # Convert the data array to NumPy and scale
    b, h, w = img.shape

    with rasterio.open(
            save_path,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=b,
            dtype=img.dtype,
            crs=src.crs,
            transform=new_transform,
            nodata=0,
            compress="zstd",
            zstd_level=13,
            interleave="band",
    ) as dst:
        dst.write(img)
