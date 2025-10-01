# import mlstac
import torch
# import cubo
import numpy as np
# import os
# import pathlib
import time
#
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm
# import torch
from pathlib import Path
import pandas as pd

import superIX.utils
from superIX.utils import parallel, sequential


if __name__ == '__main__':
    model_name = 'sr4rs'
    model = None

    # define strtification file for fps
    s2_path = Path('/data/USERS/shollend/metadata/stratification_tables/filtered/')
    data = [pd.read_csv(file) for file in s2_path.glob('*.csv')]
    s2_data = pd.concat(data)
    outpath = Path('/data/USERS/shollend/sentinel2/sr_inference')

    #outpath.mkdir(parents=True, exist_ok=True)
    save_original = False
    # Load the model
    if model_name == 'evoland':
        from superIX.evoland.utils import load_evoland, run_evoland

        IMG_BANDS = [1, 2, 3, 7, 4, 5, 6, 8, 10, 11]
        to_res = 5
        model = load_evoland(weight_path="superIX/evoland/weights/carn_3x3x64g4sw_bootstrap.onnx")

    elif model_name == 'sr4rs':
        from superIX.sr4rs.utils import load_cesbio_sr, run_sr4rs
        to_res = 2.5
        model = load_cesbio_sr(weight_path="superIX/sr4rs/weights")

    elif model_name == 'swin2_mose':
        from superIX.swin2_mose.utils import load_swin2_mose, load_config, run_swin2_mose

        device = "cuda:0"
        path = 'superIX/swin2_mose/weights/config-70.yml'
        model_weights = "superIX/swin2_mose/weights/model-70.pt"

        # load config
        cfg = load_config(path)

        # load model
        model = load_swin2_mose(model_weights, cfg)
        model.to(device)
        model.eval()

    superIX.utils.MODEL = model

    print('creating dir: ', outpath / model_name)
    Path(outpath / model_name).mkdir(parents=True, exist_ok=True)

    t1 = time.time()
    sequential(s2_data, config={'model_name': model_name,
                                'model': model,
                                'src': None,
                                'outpath': outpath,
                                'filename': None
                                })

    t2 = time.time()
    print('took [s]: ', round(t2 - t1, 2))
