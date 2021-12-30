# Digging Into Self-Supervised Learning of Feature Descriptors
This repository contains the PyTorch implementation of our work **Digging Into Self-Supervised Learning of Feature Descriptors** presented at 3DV 2021
[[Project page]](https://imelekhov.com/hndesc/)
[[ArXiv]](https://arxiv.org/abs/2110.04773)

TL;DR: The paper proposes an Unsupervised CNN-based local descriptor that is robust to illumination changes and competitve with its fully-(weakly-)supervised counterparts.

<p align="center">
  <a href="https://arxiv.org/abs/2110.04773"><img src="doc/pipeline_small_test.png" width="75%"/></a>
  <br /><em>Local image descriptors learning pipeline</em>
</p>

## Requirements
```
conda create -n hndesc_env python=3.9
conda activate hndesc_env
pip install -r requirements.txt
```
The pretrained models are available [here](https://drive.google.com/file/d/1bHJzHK6lMW424d72MpB3M6Se_tXAbSq4/view?usp=sharing). In this project, we use the [Hydra](https://hydra.cc/docs/intro/) library to handle JSON-based config files.

## Evaluation
We provide code for evaluation HNDesc on the following benchmarks/tasks: image matching (HPatches), image retrieval (rOxford5k, rParis6k, and Tokyo24/7), and camera relocalization (Aachen v1.1). The code is available under `experiments/`. Download the model [weights](https://drive.google.com/file/d/1bHJzHK6lMW424d72MpB3M6Se_tXAbSq4/view?usp=sharing) and extract them to `assets/`.

### HPatches
Once the model weights obtained, there are two ways of evaluation on HPatches:

One can run the `eval_on_hpatches.sh` script that automatically downloads the HPatches dataset and performs evaluation. Before evaluation, it is required to specify `$DATASETS_PATH` where the dataset is going to be downloaded.

Manually change the config files:
  - Open `experiments/configs/main.yaml` and change the following keys:
    - `defaults.task` to `hpatches`
    - `defaults.descriptor` to `hndesc`
    - `paths.datasets_home_dir` to `$DATASETS_PATH`
  - Run `python main.py` under `experiments/`


### Image retrieval

### Camera relocalization

## Training
The data (~24 Gb) is available [here](https://drive.google.com/file/d/18Wv0XIIMEsYeUNvbLX4GExncRyilj_WG/view?usp=sharing).


## Qualitative results

### Aachen
<p align="center">
  <img src="doc/aachen1.png" width="49%" />
  <img src="doc/aachen2.png" width="49%" />
</p>

### InLOC
<p align="center">
  <img src="doc/inloc1.png" width="49%" />
  <img src="doc/inloc2.png" width="49%" />
</p>


## Cite
If you find our work useful, please cite *both* papers:
```bibtex
  @inproceedings{Melekhov2021hardnet,
    title = {Digging Into Self-Supervised Learning of Feature Descriptors},
    author = {Melekhov, Iaroslav and Laskar, Zakaria and Li, Xiaotian and Wang, Shuzhe and Kannala Juho},
    booktitle = {In Proceedings of the International Conference on 3D Vision (3DV)},
    year = {2021}}

  @article{Melekhov2020Nian,
    author = {{Melekhov, Iaroslav and Brostow, Gabriel J. and Kannala, Juho and Turmukhambetov, Daniyar},
    title = {Image Stylization for Robust Features},
    journal = {Arxiv preprint arXiv:2008.06959},
    year = {2020}}
```
