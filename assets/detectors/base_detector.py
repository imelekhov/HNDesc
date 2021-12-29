import numpy as np
import cv2
import torch


class _DetectorBase:
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def cv2torch(cv_kpt: cv2.KeyPoint, pnt_3d: bool) -> torch.Tensor:
        pnt = (
            torch.ones((3, 1), dtype=torch.float)
            if pnt_3d
            else torch.ones((2, 1), dtype=torch.float)
        )
        pnt[0] = cv_kpt.pt[1]
        pnt[1] = cv_kpt.pt[0]
        return pnt

    @staticmethod
    def get_dummy_kpts(h, w, n_kpts, xyz=True):
        x = (
            torch.FloatTensor(np.random.randint(0, h, n_kpts))
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        y = (
            torch.FloatTensor(np.random.randint(0, w, n_kpts))
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        kpts = torch.cat((x, y), dim=1)
        if xyz:
            z = (
                torch.ones(n_kpts, dtype=torch.float)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            kpts = torch.cat((x, y, z), dim=1)
        return kpts

    def detect_img(self, img_fname: str) -> torch.Tensor:
        raise NotImplementedError

    def detect_crop(self, crop: np.ndarray) -> torch.Tensor:
        raise NotImplementedError
