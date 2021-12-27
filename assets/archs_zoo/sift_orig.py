import numpy as np
from PIL import Image
import itertools
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class SIFTModel(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def img_preprocessing(fname, device, resize_max=None, bbxs=None, resize_480x640=False):
        img_frame = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

        if resize_480x640:
            img_frame = cv2.resize(img_frame,
                                   (640, 480),
                                   interpolation=cv2.INTER_LINEAR)

        if bbxs is not None:
            img_cv2_pil = Image.fromarray(img_frame, 'L')
            img_frame = np.asarray(img_cv2_pil.crop(bbxs))

        size = img_frame.shape[:2][::-1]
        w, h = size

        size_new = [w, h]
        if resize_max and max(w, h) > resize_max:
            scale = resize_max / max(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            img_frame = cv2.resize(img_frame,
                                   (w_new, h_new),
                                   interpolation=cv2.INTER_LINEAR)
            size_new = [w_new, h_new]

        data = {'net_input': img_frame,
                'original_size': np.array(size),
                'new_size': np.array(size_new)}
        return data

    def get_resize_max_img(self):
        return None

    def forward(self, img):
        # using sift keypoints
        sift = cv2.xfeatures2d.SIFT_create(4000)
        kpts, descs = sift.detectAndCompute(img, None)
        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
        coord = torch.from_numpy(kpts).float()
        descs = torch.from_numpy(descs).float()

        # compute Root-SIFT
        descs_root_sift = F.normalize(torch.sqrt(F.normalize(descs, p=1)))

        return {
            'keypoints': coord,
            'scores': None,
            'descriptors': descs_root_sift,
            'dense_descriptors': None
        }
