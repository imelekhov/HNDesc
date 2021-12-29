import itertools
import sys

import kornia as K
from kornia.geometry.transform import imgwarp
import numpy as np
import os
from os import path as osp
import pickle
from PIL import Image
import random
import torch
from torch.utils.data import Dataset

from data.utils import HomographyAugmenter
from assets.detectors.base_detector import _DetectorBase


class MegaDepthDataset(Dataset):
    def __init__(
        self,
        img_path,
        kpts_path,
        crop_size=256,
        win_size=3,
        global_desc_dict=None,
        st_path=None,
        transforms=None,
    ):
        self.img_path = img_path
        self.kpts_path = kpts_path
        self.h_crop, self.w_crop = (crop_size, crop_size)
        self.win_size = win_size
        self.global_desc_dict = global_desc_dict
        self.st_path = st_path
        self.stylized_threshold = 0.3
        self.n_samples = 5
        self.n_kpts_crop_max = 200
        self.transforms = transforms
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Synthetic homography class
        self.homography = HomographyAugmenter(
            crop_hw=(self.h_crop, self.w_crop)
        )

        self.fnames = []
        for (dirpath, dirnames, filenames) in os.walk(self.img_path):
            for fname in filenames:
                if fname.endswith(".jpg") and fname[0] != ".":
                    self.fnames.append(osp.join(dirpath, fname))

    def _is_mask_valid(self, grid_norm):
        # the mask
        mask = (
            (grid_norm[0, :, :, 0] >= -1.0)
            & (grid_norm[0, :, :, 0] <= 1.0)
            & (grid_norm[0, :, :, 1] >= -1.0)
            & (grid_norm[0, :, :, 1] <= 1.0)
        )
        valid = not ((mask == 0).sum() == self.h_crop * self.w_crop)
        return valid

    def _generate_crops(self, img, kpts, img_st=None):
        w, h = img.size
        (
            cv_img2crop1,
            _,
            h2img,
            _,
            crop_center,
        ) = self.homography.get_random_homography(image_hw=(h, w))
        cv_img2crop2, _, _, h2crop2, _ = self.homography.get_random_homography(
            image_hw=(h, w), crop_center=crop_center
        )

        img2 = img_st if img_st is not None else img

        if kpts is not None:
            crop1, kpts1_w = self.homography.warp_image_and_kpts(
                img, kpts, cv_img2crop1
            )
            crop2, kpts2_w = self.homography.warp_image_and_kpts(
                img2, kpts, cv_img2crop2
            )
        else:
            crop1 = self.homography.warp_image(img, cv_img2crop1)
            crop2 = self.homography.warp_image(img2, cv_img2crop2)
            kpts1_w, kpts2_w = None, None

        a_H_b = (
            torch.from_numpy(np.matmul(h2img, h2crop2)).float().unsqueeze(0)
        )
        b_H_a = torch.inverse(a_H_b)
        return crop1, crop2, kpts1_w, kpts2_w, a_H_b, b_H_a

    def _get_random_crops(self, img, kpts_fname, img_st=None):
        done = False
        n_iter = 0
        stop_search_patch_niter = 10
        std_th = 20

        with open(kpts_fname, "rb") as f:
            kpts = torch.from_numpy(pickle.load(f))

        if len(kpts.shape) == 3:
            kpts = kpts.squeeze()

        while not done and (n_iter < stop_search_patch_niter):
            crop1, crop2, kpts1, kpts2, a_H_b, b_H_a = self._generate_crops(
                img, kpts, img_st
            )
            if len(kpts1) == 0:
                n_iter += 1
                continue

            if crop1.mean(axis=-1).flatten().std() < std_th:
                n_iter += 1
                continue

            a_grid_px_b = imgwarp.warp_grid(
                K.utils.create_meshgrid(self.h_crop, self.w_crop, False), a_H_b
            )
            b_grid_px_a = imgwarp.warp_grid(
                K.utils.create_meshgrid(self.h_crop, self.w_crop, False), b_H_a
            )

            # normalize coordinates and compute the valid mask
            a_grid_norm_b = K.geometry.conversions.normalize_pixel_coordinates(
                a_grid_px_b, self.h_crop, self.w_crop
            )
            b_grid_norm_a = K.geometry.conversions.normalize_pixel_coordinates(
                b_grid_px_a, self.h_crop, self.w_crop
            )

            if (not self._is_mask_valid(a_grid_norm_b)) or (
                not self._is_mask_valid(b_grid_norm_a)
            ):
                n_iter += 1
                continue

            done = True

        if not done:
            crop1, crop2, _, _, _, _ = self._generate_crops(img, None)
            kpts1 = _DetectorBase.get_dummy_kpts(
                self.h_crop, self.w_crop, self.n_kpts_crop_max
            ).squeeze()
            a_H_b = torch.eye(3).unsqueeze(0)
            b_H_a = a_H_b.clone()
            a_grid_px_b = imgwarp.warp_grid(
                K.utils.create_meshgrid(self.h_crop, self.w_crop, False), a_H_b
            )
            b_grid_px_a = imgwarp.warp_grid(
                K.utils.create_meshgrid(self.h_crop, self.w_crop, False), b_H_a
            )

        kpts = torch.stack([kpt[:2] for kpt in kpts])

        return {
            "crop_src": crop1,
            "crop_trg": crop2,
            "kpts_crop_src": kpts1,
            "kpts_img": kpts,
            "crop_size": self.h_crop,
            "a_grid_px_b": a_grid_px_b.squeeze(),
            "b_grid_px_a": b_grid_px_a.squeeze(),
            "a_H_b": a_H_b.squeeze(),
            "b_H_a": b_H_a.squeeze(),
        }

    @staticmethod
    def _get_target_kpts(kpts_src, h_mtx):
        kpts_trg = torch.matmul(h_mtx, kpts_src.t()).t()
        kpts_trg = [kpt[:2] / kpt[-1] for kpt in kpts_trg]
        kpts_trg = torch.stack(kpts_trg)
        return kpts_trg

    def _get_kpt_fname(self, img_fname):
        pathname_out = osp.join(
            self.kpts_path, img_fname[len(self.img_path) + 1 :]
        )
        pathname_out = pathname_out[: pathname_out.rindex("/")]
        kpt_fname = osp.join(
            pathname_out, img_fname.split("/")[-1][:-4] + ".pkl"
        )
        return kpt_fname

    def _get_global_desc(self, img_fname):
        pathname_out = osp.join(
            self.global_desc_path, img_fname[len(self.img_path) + 1 :]
        )
        pathname_out = pathname_out[: pathname_out.rindex("/")]
        global_desc_fname = osp.join(
            pathname_out, img_fname.split("/")[-1][:-4] + ".pkl"
        )
        with open(global_desc_fname, "rb") as f:
            gdesc = torch.from_numpy(pickle.load(f))
        return gdesc

    def _is_kpt_valid(self, kpt):
        return (0 <= kpt[0] <= self.w_crop - 1) and (
            0 <= kpt[1] <= self.h_crop - 1
        )

    def __getitem__(self, item):
        img_fname = self.fnames[item]
        kpt_fname = self._get_kpt_fname(img_fname)

        img = Image.open(img_fname).convert("RGB")

        img_st = None
        if self.st_path is not None:
            if random.uniform(0, 1) > self.stylized_threshold:
                img_st_fname = osp.join(
                    self.st_path, img_fname[len(self.img_path) + 1 :]
                )

                # double check: if the stylized version does not exist for some reason...
                if not osp.exists(img_st_fname):
                    img_st = img
                else:
                    img_st = Image.open(img_st_fname).convert("RGB")

        crops_data = self._get_random_crops(img, kpt_fname, img_st=img_st)

        kpts_crop_src = crops_data["kpts_crop_src"]

        kpts_crop_trg = self._get_target_kpts(
            kpts_crop_src, crops_data["b_H_a"]
        )
        kpts_crop_src = torch.stack([kpt[:2] for kpt in kpts_crop_src])
        mask_consistency = torch.BoolTensor(
            [self._is_kpt_valid(kpt) for kpt in kpts_crop_trg]
        )

        crop1 = crops_data["crop_src"]
        crop2 = crops_data["crop_trg"]

        if sum(mask_consistency) == 0:
            kpts_crop_trg = kpts_crop_src.clone()
            crop2 = crop1
        else:
            kpts_crop_src = kpts_crop_src[mask_consistency]
            kpts_crop_trg = kpts_crop_trg[mask_consistency]

        assert (
            kpts_crop_src.shape[0] == kpts_crop_trg.shape[0]
        ), "Number of kpts in src and trg should be equal"

        # Let us pad the number of keypoints up to self.kpts_max to be able to use default collate_fn
        kpts_crop_src_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
        kpts_crop_trg_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
        mask_kpts_collate = torch.BoolTensor(self.n_kpts_crop_max, 1).fill_(0)

        if len(kpts_crop_src) > self.n_kpts_crop_max:
            kpts_crop_src = kpts_crop_src[: self.n_kpts_crop_max]
            kpts_crop_trg = kpts_crop_trg[: self.n_kpts_crop_max]

        kpts_crop_src_pad[: len(kpts_crop_src)] = kpts_crop_src
        kpts_crop_trg_pad[: len(kpts_crop_src)] = kpts_crop_trg
        mask_kpts_collate[: len(kpts_crop_src)] = True

        def kpts_win(kpts, window):
            kpts_out = []
            kpts_r = torch.round(kpts).long()
            for kpt in kpts_r:
                if (
                    0 <= kpt[0] - 1
                    or kpt[0] + 1 < self.h_crop
                    or 0 <= kpt[1] - 1
                    or kpt[1] + 1 < self.w_crop
                ):
                    win_kpts = torch.zeros(
                        (win.shape[0] ** 2, 2), dtype=torch.int
                    )
                    win_kpts[:, 0], win_kpts[:, 1] = kpt[0], kpt[1]
                else:
                    win_kpts = kpt.repeat(window.shape[0], 1) + window.to(kpt)
                    win_kpts = list(
                        itertools.product(
                            win_kpts[:, 0].numpy(), win_kpts[:, 1].numpy()
                        )
                    )
                    win_kpts = torch.from_numpy(np.array([*win_kpts]))
                kpts_out.append(win_kpts)
            kpts_out = torch.stack(kpts_out).view(-1, 2)
            return kpts_out

        # let us associate each keypoint with a window
        win = torch.arange(-1, 2).unsqueeze(1)
        kpts_src_win = kpts_win(kpts_crop_src_pad, win)
        kpts_trg_win = kpts_win(kpts_crop_trg_pad, win)

        crop_hn, kpts_hn, kpts_hn_pad, mask_kpts_hn_collate = [], [], [], []
        # let us find hard negative image based on global representation
        if self.global_desc_dict is not None:
            img_fullname = self.fnames[item]
            img_fname = img_fullname[len(self.img_path) + 1 :]

            with open(self.global_desc_dict, "rb") as f:
                hn_fnames_dict = pickle.load(f)

            split = self.img_path.split("/")[-1]  # train/test

            knn_id = 0
            done = False
            while not done:
                img_fname_hn = hn_fnames_dict[split + "/" + img_fname][knn_id]
                img_fname_hn = img_fname_hn[len(split) + 1 :]
                img_hn = Image.open(
                    osp.join(self.img_path, img_fname_hn)
                ).convert("RGB")
                img_hn_arr = np.array(img_hn)
                if (
                    img_hn_arr.shape[0] > self.h_crop
                    and img_hn_arr.shape[1] > self.w_crop
                ):
                    done = True
                knn_id += 1

            kpts_hn_fname = osp.join(
                self.kpts_path, img_fname_hn[:-4] + ".pkl"
            )
            with open(kpts_hn_fname, "rb") as f:
                kpts_hn = torch.from_numpy(pickle.load(f))

            # determine crop size
            output_size_a = min(img_hn.size, (self.h_crop, self.w_crop))

            mask_kpts_hn = np.zeros(
                (img_hn_arr.shape[0], img_hn_arr.shape[1]), dtype=np.int
            )
            val = 1
            for row_id, col_id in zip(
                kpts_hn[:, 1].int(), kpts_hn[:, 0].int()
            ):
                mask_kpts_hn[row_id, col_id] = val
                val += 1

            mask_kpts_hn_bool = mask_kpts_hn > 0

            def window1(x, size, w):
                l = x - int(0.5 + size / 2)
                r = l + int(0.5 + size)
                if l < 0:
                    l, r = (0, r - l)
                if r > w:
                    l, r = (l + w - r, w)
                if l < 0:
                    l, r = 0, w  # larger than width
                return slice(l, r)

            def window(cx, cy, win_size, scale, img_shape):
                return (
                    window1(cy, win_size[1] * scale, img_shape[0]),
                    window1(cx, win_size[0] * scale, img_shape[1]),
                )

            n_valid_pixel = mask_kpts_hn_bool.sum()
            sample_w = (
                mask_kpts_hn_bool.astype(np.float)
                / (1e-16 + n_valid_pixel.astype(np.float))
            ).astype(np.float)

            def sample_valid_pixel():
                p = sample_w.ravel()
                p = p * (1.0 / (1e-16 + p.sum()))
                try:
                    n = np.random.choice(sample_w.size, p=p)
                except:
                    n = np.random.choice(sample_w.size)
                y, x = np.unravel_index(n, sample_w.shape)
                return x, y

            # Find suitable left and right windows
            trials = 0  # take the best out of few trials
            best = -np.inf, None
            for i in range(50 * self.n_samples):
                if trials >= self.n_samples:
                    break  # finished!

                # pick a random valid point from the first image
                if n_valid_pixel == 0:
                    break
                c1x, c1y = sample_valid_pixel()  # c1x_arr[i], c1y_arr[i]

                win1 = window(c1x, c1y, output_size_a, 1, img_hn_arr.shape)

                score1 = mask_kpts_hn_bool[win1].ravel().mean()
                score = score1

                trials += 1
                if score > best[0]:
                    best = score, win1

            if None in best:  # counldn't find a good windows
                c1x, c1y = sample_valid_pixel()  # c1x_arr[0], c1y_arr[0] #(1)
                win1 = window(c1x, c1y, output_size_a, 1, img_hn_arr.shape)
                img_a = img_hn_arr[win1]
                mask_kpts_bool = mask_kpts_hn_bool[win1]
            else:
                win1 = best[1]
                img_a = img_hn_arr[win1]
                mask_kpts_bool = mask_kpts_hn_bool[win1]

            row_kpts_src, col_kpts_src = np.nonzero(mask_kpts_bool)
            kpts_hn = torch.IntTensor([row_kpts_src, col_kpts_src]).t()
            # kpts_hn = torch.IntTensor([col_kpts_src, row_kpts_src]).t()

            # let us create padded arrays
            kpts_hn_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
            mask_kpts_hn_collate = torch.BoolTensor(
                self.n_kpts_crop_max, 1
            ).fill_(0)

            n_valid_kpts = kpts_hn.shape[0]

            if n_valid_kpts > self.n_kpts_crop_max:
                kpts_hn = kpts_hn[: self.n_kpts_crop_max, :]

            kpts_hn_pad[:n_valid_kpts] = kpts_hn
            mask_kpts_hn_collate[:n_valid_kpts] = True

            if self.transforms:
                crop_hn = self.transforms(image=img_a)["image"]

        if self.transforms:
            crop1 = self.transforms(image=crop1)["image"]
            crop2 = self.transforms(image=crop2)["image"]

        if isinstance(crop_hn, list):
            crop_hn = torch.FloatTensor(crop_hn)

        return {
            "img_fname": self.fnames[item],
            "crop_src": crop1,
            "crop_trg": crop2,
            "crop_hn": crop_hn,
            "h_mtx": crops_data["b_H_a"],
            "crop_src_kpts": kpts_crop_src_pad,
            "crop_trg_kpts": kpts_crop_trg_pad,
            "crop_kpts_hn": kpts_hn_pad,
            "kpts_src_win": kpts_src_win,
            "kpts_trg_win": kpts_trg_win,
            "mask_valid_kpts": mask_kpts_collate,
            "mask_valid_kpts_hn": mask_kpts_hn_collate,
        }

    def __len__(self):
        return len(self.fnames)


class PrecomputedValidationDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.fnames = [fname for fname in os.listdir(self.path)]

    def __getitem__(self, item):
        fname = self.fnames[item]
        with open(osp.join(self.path, fname), "rb") as f:
            metadata = pickle.load(f)

        return metadata

    def __len__(self):
        return len(self.fnames)
