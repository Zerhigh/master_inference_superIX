# Repository: master_inference_superIX

This repository is part of the master’s thesis: [Evaluating_Sentinel-2_Super-Resolution_Algorithms_for_Automated_Building_Delineation](https://github.com/Zerhigh/Evaluating_Sentinel-2_Super-Resolution_Algorithms_for_Automated_Building_Delineation)

This project contains scripts and utilities for super-resolving (and interpolating) Sentinel-2 imagery with the superIX framework [https://huggingface.co/isp-uv-es/superIX/tree/main](https://huggingface.co/isp-uv-es/superIX/tree/main).

The following models can be used:
- Evoland
- SR4RS
- Swin2Mose

Access the corresponding weights from superIX and insert into the corresponding folders in `superIX/*/weights/`. The code in this repository allows the SR and subsequent interpolation of all model outputs to images with a resolution of 2.5m and image shapes of `(4, 512, 512)`.

The repository includes:

- `modify_inferred.py` — sample utilities for reordering image bands and adding compressing options for geotiff drivers   
- `s2_inference.py` — main script used to initialise the model inference of the selected models.

---
