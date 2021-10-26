import numpy as np
import cv2
import torch
import random
import time
from assets.detectors.base_detector import _DetectorBase


class ORBdetector(_DetectorBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.N_KPTS_IMG = self.cfg['n_kpts_img']
        self.N_KPTS_CROP = self.cfg['n_kpts_crop']
        self.detector = cv2.ORB_create(self.N_KPTS_IMG)

    def detect_img(self, img: np.ndarray) -> torch.Tensor:
        t1 = time.time()
        kpts = self.detector.detect(img, None)
        kpts_torch = []

        for cv_kpt in kpts:
            kpts_torch.append(self.cv2torch(cv_kpt, pnt_3d=True))

        if not kpts_torch:
            print('Could not detect any keypoints -> generate dummy keypoints')
            kpts_torch = self.get_dummy_kpts(img.shape[0], img.shape[1], self.N_KPTS_CROP)
        else:
            kpts_torch = torch.stack(kpts_torch)

        return kpts_torch

    def detect_crop(self, crop: np.ndarray) -> torch.Tensor:
        kpts = self.detect_img(crop)
        n_kpts = kpts.shape[0] if kpts.shape[0] < self.N_KPTS_CROP else self.N_KPTS_CROP
        kpts_crop = torch.FloatTensor(random.sample(kpts.tolist(), n_kpts))
        return kpts_crop
