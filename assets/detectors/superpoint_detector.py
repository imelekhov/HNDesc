import numpy as np
import cv2
import torch
from assets.archs_zoo.superpoint_orig import SuperPoint
from assets.detectors.base_detector import _DetectorBase


class Superpointdetector(_DetectorBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.N_KPTS_IMG = self.cfg['n_kpts_img']
        self.N_KPTS_CROP = self.cfg['n_kpts_crop']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = SuperPoint(cfg).to(self.device)

    @staticmethod
    def frame2tensor(frame, device):
        return torch.from_numpy(frame / 255.).float()[None, None].to(device)

    @staticmethod
    def rgb2gray(rgb_crop):
        return np.dot(rgb_crop[..., :3], [0.2989, 0.5870, 0.1140])

    def _detect(self, arr: np.ndarray, xyz=True) -> torch.Tensor:
        with torch.no_grad():
            out = self.detector(self.frame2tensor(arr, self.device))
            kpts_torch = out['keypoints'][0]
            if len(kpts_torch) == 0:
                print('Could not detect any keypoints -> generate dummy keypoints')
                kpts_torch = self.get_dummy_kpts(arr.shape[1], arr.shape[0], self.N_KPTS_CROP, xyz)
            else:
                if xyz:
                    ones = torch.ones((kpts_torch.shape[0], 1), dtype=torch.float).to(self.device)
                    kpts_torch = torch.cat((kpts_torch, ones), dim=-1)
        return kpts_torch

    def detect_img(self, img_fname: str, xyz=True) -> torch.Tensor:
        img_frame = cv2.imread(img_fname, cv2.IMREAD_GRAYSCALE)
        kpts = self._detect(img_frame, xyz)
        return kpts

    def detect_crop(self, crop: np.ndarray, xyz=True) -> torch.Tensor:
        crop_grey = self.rgb2gray(crop)
        kpts = self._detect(crop_grey, xyz)
        return kpts
