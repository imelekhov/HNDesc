import numpy as np
import torch
from assets.detectors.orb_detector import ORBdetector
from assets.detectors.superpoint_detector import Superpointdetector


class LocalDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.detector = self._detector_factory()

    def _detector_factory(self):
        detector = None
        if self.cfg["name"] == "orb":
            detector = ORBdetector(self.cfg)
        elif self.cfg["name"] == "superpoint":
            detector = Superpointdetector(self.cfg)
        else:
            raise ValueError("detector name is not right")
        return detector

    def detect_img(self, img: str) -> torch.Tensor:
        return self.detector.detect_img(img)

    def detect_crop(self, crop: np.ndarray) -> torch.Tensor:
        return self.detector.detect_crop(crop)
