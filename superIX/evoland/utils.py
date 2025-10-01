import torch
import pickle
import numpy as np
import opensr_test
import onnxruntime as ort
from typing import List, Union


def load_evoland(weight_path) -> np.ndarray:
    # ONNX inference session options
    so = ort.SessionOptions()
    so.intra_op_num_threads = 10
    so.inter_op_num_threads = 10
    so.use_deterministic_compute = True

    # Execute on cpu only
    ep_list = ["CPUExecutionProvider"]
    ep_list.insert(0, "CUDAExecutionProvider")

    ort_session = ort.InferenceSession(
        weight_path,
        sess_options=so,
        providers=ep_list
    )
    ort_session.set_providers(["CPUExecutionProvider"])
    ro = ort.RunOptions()

    return [ort_session, ro]


def run_evoland(
        model: List,
        lr: np.ndarray,
        hr: np.ndarray = None
) -> dict:
    ort_session, ro = model

    # Bands to use
    # bands are re-sorted to: blue, green, red, nir, ...
    bands = [1, 2, 3, 7, 4, 5, 6, 8, 10, 11]
    lr = lr[bands]

    if lr.shape[1] == 121:
        # add padding
        lr = torch.nn.functional.pad(
            torch.from_numpy(lr[None]).float(),
            pad=(3, 4, 3, 4),
            mode='reflect'
        ).squeeze().cpu().numpy()

        # run the model
        sr = ort_session.run(
            None,
            {"input": lr[None]},
            run_options=ro
        )[0].squeeze()

        # remove padding
        sr = sr[:, 3 * 2:-4 * 2, 3 * 2:-4 * 2].astype(np.uint16)
        lr = lr[:, 3:-4, 3:-4].astype(np.uint16)
    else:
        # run the model
        sr = ort_session.run(
            None,
            {"input": lr[None].astype(np.float32)},
            run_options=ro
        )[0].squeeze()

    # Run the model
    # bands are returned as: blue, green, red, nir
    return {
        "lr": lr[[0, 1, 2, 3]],
        "sr": sr[[0, 1, 2, 3]],
    }
