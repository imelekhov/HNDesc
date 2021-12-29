# Digging Into Self-Supervised Learning of Feature Descriptors
This repository contains the PyTorch implementation of our work **Digging Into Self-Supervised Learning of Feature Descriptors** presented at 3DV 2021 
[[Project page]](https://imelekhov.com/hndesc/)
[[ArXiv]](https://arxiv.org/abs/2110.04773)

TL;DR: The paper proposes an Unsupervised CNN-based local descriptor that is robust to illumination changes and competitve with its fully-(weakly-)supervised counterparts.

<p align="center">
  <a href="https://arxiv.org/abs/2110.04773"><img src="doc/pipeline_small_test.png" width="75%"/></a>
  <br /><em>Local image descriptors learning pipeline</em>
</p>

## Abstract
Fully-supervised CNN-based approaches for learning local image descriptors have shown remarkable results in a wide range of geometric tasks. However, most of them require per-pixel ground-truth keypoint correspondence data which is difficult to acquire at scale. In this work, we focus on understanding the limitations of existing self-supervised approaches and propose a set of improvements that combined lead to powerful feature descriptors. We show that increasing the search space from in-pair to in-batch for hard negative mining brings consistent improvement. To enhance the discriminativeness of feature descriptors, we propose a coarse-to-fine method for mining local hard negatives from a wider search space by using global visual image descriptors. We demonstrate that a combination of synthetic homography transformation, color augmentation, and photorealistic image stylization produces useful representations that are viewpoint and illumination invariant. The feature descriptors learned by the proposed approach perform competitively and surpass their fully- and weakly-supervised counterparts on various geometric benchmarks such as image-based localization, sparse feature matching, and image retrieval.

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
