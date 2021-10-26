import os
from os import path as osp
import time
import random
import numpy as np
import itertools
from PIL import Image
import pickle
import torch
from torch.utils.data import Dataset
import kornia as K
from data.utils import HomographyAugmenter
from assets.detectors.base_detector import _DetectorBase


class MegaDepthDataset(Dataset):
    def __init__(self,
                 img_path,
                 kpts_path,
                 crop_size=256,
                 win_size=3,
                 global_desc_dict=None,
                 st_path=None,
                 transforms=None):
        self.img_path = img_path
        self.kpts_path = kpts_path
        self.h_crop, self.w_crop = (crop_size, crop_size)
        self.win_size = win_size
        self.global_desc_dict = global_desc_dict
        self.st_path = st_path
        self.stylized_threshold = 0.3 # 0.5
        self.n_samples = 5
        self.n_kpts_crop_max = 200
        self.transforms = transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Synthetic homography class
        self.homography = HomographyAugmenter(crop_hw=(self.h_crop, self.w_crop))

        self.fnames = []
        for (dirpath, dirnames, filenames) in os.walk(self.img_path):
            for fname in filenames:
                if fname.endswith('.jpg') and fname[0] != '.':
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
        cv_img2crop1, _, h2img, _, crop_center = self.homography.get_random_homography(image_hw=(h, w))
        cv_img2crop2, _, _, h2crop2, _ = self.homography.get_random_homography(image_hw=(h, w),
                                                                               crop_center=crop_center)

        img2 = img_st if img_st is not None else img

        if kpts is not None:
            crop1, kpts1_w = self.homography.warp_image_and_kpts(img, kpts, cv_img2crop1)
            crop2, kpts2_w = self.homography.warp_image_and_kpts(img2, kpts, cv_img2crop2)
        else:
            crop1 = self.homography.warp_image(img, cv_img2crop1)
            crop2 = self.homography.warp_image(img2, cv_img2crop2)
            kpts1_w, kpts2_w = None, None

        a_H_b = torch.from_numpy(np.matmul(h2img, h2crop2)).float().unsqueeze(0)
        b_H_a = torch.inverse(a_H_b)
        return crop1, crop2, kpts1_w, kpts2_w, a_H_b, b_H_a

    def _get_random_crops(self, img, kpts_fname, img_st=None):
        done = False
        n_iter = 0
        stop_search_patch_niter = 10
        std_th = 20

        with open(kpts_fname, 'rb') as f:
            kpts = torch.from_numpy(pickle.load(f))

        if len(kpts.shape) == 3:
            kpts = kpts.squeeze()

        while not done and (n_iter < stop_search_patch_niter):
            crop1, crop2, kpts1, kpts2, a_H_b, b_H_a = self._generate_crops(img, kpts, img_st)
            if len(kpts1) == 0:
                n_iter += 1
                continue

            if crop1.mean(axis=-1).flatten().std() < std_th:
                n_iter += 1
                continue

            a_grid_px_b = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    a_H_b)
            b_grid_px_a = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    b_H_a)

            # normalize coordinates and compute the valid mask
            a_grid_norm_b = K.geometry.conversions.normalize_pixel_coordinates(a_grid_px_b, self.h_crop, self.w_crop)
            b_grid_norm_a = K.geometry.conversions.normalize_pixel_coordinates(b_grid_px_a, self.h_crop, self.w_crop)

            if (not self._is_mask_valid(a_grid_norm_b)) or (not self._is_mask_valid(b_grid_norm_a)):
                n_iter += 1
                continue

            done = True

        if not done:
            crop1, crop2, _, _, _, _ = self._generate_crops(img, None)
            kpts1 = _DetectorBase.get_dummy_kpts(self.h_crop, self.w_crop, self.n_kpts_crop_max).squeeze()
            a_H_b = torch.eye(3).unsqueeze(0)
            b_H_a = a_H_b.clone()
            a_grid_px_b = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    a_H_b)
            b_grid_px_a = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    b_H_a)

        kpts = torch.stack([kpt[:2] for kpt in kpts])

        return {"crop_src": crop1,
                "crop_trg": crop2,
                "kpts_crop_src": kpts1,
                "kpts_img": kpts,
                "crop_size": self.h_crop,
                "a_grid_px_b": a_grid_px_b.squeeze(),
                "b_grid_px_a": b_grid_px_a.squeeze(),
                "a_H_b": a_H_b.squeeze(),
                "b_H_a": b_H_a.squeeze()}

    @staticmethod
    def _get_target_kpts(kpts_src, h_mtx):
        kpts_trg = torch.matmul(h_mtx, kpts_src.t()).t()
        kpts_trg = [kpt[:2] / kpt[-1] for kpt in kpts_trg]
        kpts_trg = torch.stack(kpts_trg)
        return kpts_trg

    def _get_kpt_fname(self, img_fname):
        pathname_out = osp.join(self.kpts_path, img_fname[len(self.img_path) + 1:])
        pathname_out = pathname_out[:pathname_out.rindex('/')]
        kpt_fname = osp.join(pathname_out, img_fname.split('/')[-1][:-4] + '.pkl')
        return kpt_fname

    def _get_global_desc(self, img_fname):
        pathname_out = osp.join(self.global_desc_path, img_fname[len(self.img_path) + 1:])
        pathname_out = pathname_out[:pathname_out.rindex('/')]
        global_desc_fname = osp.join(pathname_out, img_fname.split('/')[-1][:-4] + '.pkl')
        with open(global_desc_fname, 'rb') as f:
            gdesc = torch.from_numpy(pickle.load(f))
        return gdesc

    def _is_kpt_valid(self, kpt):
        return (0 <= kpt[0] <= self.w_crop - 1) and (0 <= kpt[1] <= self.h_crop - 1)

    def __getitem__(self, item):
        img_fname = self.fnames[item]
        kpt_fname = self._get_kpt_fname(img_fname)

        img = Image.open(img_fname).convert('RGB')
        # img_greyscale = Image.open(self.fnames[item]).convert('L')

        img_st = None
        if self.st_path is not None:
            if random.uniform(0, 1) > self.stylized_threshold:
                img_st_fname = osp.join(self.st_path, img_fname[len(self.img_path)+1:])

                # double check: if the stylized version does not exist for some reason...
                if not osp.exists(img_st_fname):
                    img_st = img
                else:
                    img_st = Image.open(img_st_fname).convert('RGB')

        crops_data = self._get_random_crops(img, kpt_fname, img_st=img_st)

        kpts_crop_src = crops_data['kpts_crop_src']

        kpts_crop_trg = self._get_target_kpts(kpts_crop_src, crops_data['b_H_a'])
        kpts_crop_src = torch.stack([kpt[:2] for kpt in kpts_crop_src])
        mask_consistency = torch.BoolTensor([self._is_kpt_valid(kpt) for kpt in kpts_crop_trg])

        crop1 = crops_data['crop_src']
        crop2 = crops_data['crop_trg']

        if sum(mask_consistency) == 0:
            kpts_crop_trg = kpts_crop_src.clone()
            crop2 = crop1
        else:
            kpts_crop_src = kpts_crop_src[mask_consistency]
            kpts_crop_trg = kpts_crop_trg[mask_consistency]

        assert kpts_crop_src.shape[0] == kpts_crop_trg.shape[0], 'Number of kpts in src and trg should be equal'

        # Let us pad the number of keypoints up to self.kpts_max to be able to use default collate_fn
        kpts_crop_src_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
        kpts_crop_trg_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
        mask_kpts_collate = torch.BoolTensor(self.n_kpts_crop_max, 1).fill_(0)

        if len(kpts_crop_src) > self.n_kpts_crop_max:
            kpts_crop_src = kpts_crop_src[:self.n_kpts_crop_max]
            kpts_crop_trg = kpts_crop_trg[:self.n_kpts_crop_max]

        kpts_crop_src_pad[:len(kpts_crop_src)] = kpts_crop_src
        kpts_crop_trg_pad[:len(kpts_crop_src)] = kpts_crop_trg
        mask_kpts_collate[:len(kpts_crop_src)] = True

        def kpts_win(kpts, window):
            kpts_out = []
            kpts_r = torch.round(kpts).long()
            for kpt in kpts_r:
                if 0 <= kpt[0] - 1 or kpt[0] + 1 < self.h_crop or 0 <= kpt[1] - 1 or kpt[1] + 1 < self.w_crop:
                    win_kpts = torch.zeros((win.shape[0]**2, 2), dtype=torch.int)
                    win_kpts[:, 0], win_kpts[:, 1] = kpt[0], kpt[1]
                else:
                    win_kpts = kpt.repeat(window.shape[0], 1) + window.to(kpt)
                    win_kpts = list(itertools.product(win_kpts[:, 0].numpy(), win_kpts[:, 1].numpy()))
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
            img_fname = img_fullname[len(self.img_path) + 1:]

            with open(self.global_desc_dict, 'rb') as f:
                hn_fnames_dict = pickle.load(f)

            split = self.img_path.split('/')[-1]  # train/test

            knn_id = 0
            done = False
            while not done:
                img_fname_hn = hn_fnames_dict[split + '/' + img_fname][knn_id]
                img_fname_hn = img_fname_hn[len(split) + 1:]
                img_hn = Image.open(osp.join(self.img_path, img_fname_hn)).convert('RGB')
                img_hn_arr = np.array(img_hn)
                if img_hn_arr.shape[0] > self.h_crop and img_hn_arr.shape[1] > self.w_crop:
                    done = True
                knn_id += 1

            kpts_hn_fname = osp.join(self.kpts_path, img_fname_hn[:-4] + '.pkl')
            with open(kpts_hn_fname, 'rb') as f:
                kpts_hn = torch.from_numpy(pickle.load(f))

            # determine crop size
            output_size_a = min(img_hn.size, (self.h_crop, self.w_crop))

            mask_kpts_hn = np.zeros((img_hn_arr.shape[0], img_hn_arr.shape[1]), dtype=np.int)
            val = 1
            for row_id, col_id in zip(kpts_hn[:, 1].int(), kpts_hn[:, 0].int()):
                mask_kpts_hn[row_id, col_id] = val
                val += 1

            mask_kpts_hn_bool = mask_kpts_hn > 0

            def window1(x, size, w):
                l = x - int(0.5 + size / 2)
                r = l + int(0.5 + size)
                if l < 0: l, r = (0, r - l)
                if r > w: l, r = (l + w - r, w)
                if l < 0: l, r = 0, w  # larger than width
                return slice(l, r)

            def window(cx, cy, win_size, scale, img_shape):
                return (window1(cy, win_size[1] * scale, img_shape[0]),
                        window1(cx, win_size[0] * scale, img_shape[1]))

            n_valid_pixel = mask_kpts_hn_bool.sum()
            sample_w = (mask_kpts_hn_bool.astype(np.float) / (1e-16 + n_valid_pixel.astype(np.float))).astype(np.float)

            def sample_valid_pixel():
                p = sample_w.ravel()
                p = p * (1. / (1e-16 + p.sum()))
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
                if trials >= self.n_samples: break  # finished!

                # pick a random valid point from the first image
                if n_valid_pixel == 0: break
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
            mask_kpts_hn_collate = torch.BoolTensor(self.n_kpts_crop_max, 1).fill_(0)

            n_valid_kpts = kpts_hn.shape[0]

            if n_valid_kpts > self.n_kpts_crop_max:
                kpts_hn = kpts_hn[:self.n_kpts_crop_max, :]

            kpts_hn_pad[:n_valid_kpts] = kpts_hn
            mask_kpts_hn_collate[:n_valid_kpts] = True

            if self.transforms:
                crop_hn = self.transforms(image=img_a)['image']

        if self.transforms:
            crop1 = self.transforms(image=crop1)['image']
            crop2 = self.transforms(image=crop2)['image']

        return {'img_fname': self.fnames[item],
                'crop_src': crop1,
                'crop_trg': crop2,
                'crop_hn': crop_hn,
                'h_mtx': crops_data['b_H_a'],
                'crop_src_kpts': kpts_crop_src_pad,
                'crop_trg_kpts': kpts_crop_trg_pad,
                'crop_kpts_hn': kpts_hn_pad,
                'kpts_src_win': kpts_src_win,
                'kpts_trg_win': kpts_trg_win,
                'mask_valid_kpts': mask_kpts_collate,
                'mask_valid_kpts_hn': mask_kpts_hn_collate}

    def __len__(self):
        return len(self.fnames)


class MegaDepthPhototourismDataset(Dataset):
    def __init__(self,
                 img_path_m,
                 img_path_p,
                 kpts_path_m,
                 kpts_path_p,
                 global_desc_dict_m=None,
                 global_desc_dict_p=None,
                 crop_size=192,
                 win_size=3,
                 st_path_m=None,
                 st_path_p=None,
                 transforms=None):
        self.img_path_m = img_path_m
        self.img_path_p = img_path_p
        self.kpts_path_m = kpts_path_m
        self.kpts_path_p = kpts_path_p
        self.global_desc_dict_m = global_desc_dict_m
        self.global_desc_dict_p = global_desc_dict_p
        self.h_crop, self.w_crop = (crop_size, crop_size)
        self.win_size = win_size
        self.st_path_m = st_path_m
        self.st_path_p = st_path_p
        self.stylized_threshold = 0.5
        self.n_kpts_crop_max = 200
        self.n_samples = 5
        self.transforms = transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Synthetic homography class
        self.homography = HomographyAugmenter(crop_hw=(self.h_crop, self.w_crop))

        self.fnames_p = []
        self.fnames_m = []
        for paths in [self.img_path_m, self.img_path_p]:
            for (dirpath, dirnames, filenames) in os.walk(paths):
                for fname in filenames:
                    if fname.endswith('.jpg') and fname[0] != '.':
                        if 'phototourism' in dirpath:
                            self.fnames_p.append(osp.join(dirpath, fname))
                        elif 'MegaDepth' in dirpath:
                            self.fnames_m.append(osp.join(dirpath, fname))

        self.fnames = self.fnames_m + self.fnames_p

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
        cv_img2crop1, _, h2img, _, crop_center = self.homography.get_random_homography(image_hw=(h, w))
        cv_img2crop2, _, _, h2crop2, _ = self.homography.get_random_homography(image_hw=(h, w),
                                                                               crop_center=crop_center)

        img2 = img_st if img_st is not None else img

        if kpts is not None:
            crop1, kpts1_w = self.homography.warp_image_and_kpts(img, kpts, cv_img2crop1)
            crop2, kpts2_w = self.homography.warp_image_and_kpts(img2, kpts, cv_img2crop2)
        else:
            crop1 = self.homography.warp_image(img, cv_img2crop1)
            crop2 = self.homography.warp_image(img2, cv_img2crop2)
            kpts1_w, kpts2_w = None, None

        a_H_b = torch.from_numpy(np.matmul(h2img, h2crop2)).float().unsqueeze(0)
        b_H_a = torch.inverse(a_H_b)
        return crop1, crop2, kpts1_w, kpts2_w, a_H_b, b_H_a

    def _get_random_crops(self, img, kpts_fname, img_st=None):
        done = False
        n_iter = 0
        stop_search_patch_niter = 10
        std_th = 20

        with open(kpts_fname, 'rb') as f:
            kpts = torch.from_numpy(pickle.load(f))

        if len(kpts.shape) == 3:
            kpts = kpts.squeeze()

        while not done and (n_iter < stop_search_patch_niter):
            crop1, crop2, kpts1, kpts2, a_H_b, b_H_a = self._generate_crops(img, kpts, img_st)
            if len(kpts1) == 0:
                n_iter += 1
                continue

            if crop1.mean(axis=-1).flatten().std() < std_th:
                n_iter += 1
                continue

            a_grid_px_b = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    a_H_b)
            b_grid_px_a = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    b_H_a)

            # normalize coordinates and compute the valid mask
            a_grid_norm_b = K.geometry.conversions.normalize_pixel_coordinates(a_grid_px_b, self.h_crop, self.w_crop)
            b_grid_norm_a = K.geometry.conversions.normalize_pixel_coordinates(b_grid_px_a, self.h_crop, self.w_crop)

            if (not self._is_mask_valid(a_grid_norm_b)) or (not self._is_mask_valid(b_grid_norm_a)):
                n_iter += 1
                continue

            done = True

        if not done:
            crop1, crop2, _, _, _, _ = self._generate_crops(img, None)
            kpts1 = _DetectorBase.get_dummy_kpts(self.h_crop, self.w_crop, self.n_kpts_crop_max).squeeze()
            a_H_b = torch.eye(3).unsqueeze(0)
            b_H_a = a_H_b.clone()
            a_grid_px_b = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    a_H_b)
            b_grid_px_a = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    b_H_a)

        kpts = torch.stack([kpt[:2] for kpt in kpts])

        return {"crop_src": crop1,
                "crop_trg": crop2,
                "kpts_crop_src": kpts1,
                "kpts_img": kpts,
                "crop_size": self.h_crop,
                "a_grid_px_b": a_grid_px_b.squeeze(),
                "b_grid_px_a": b_grid_px_a.squeeze(),
                "a_H_b": a_H_b.squeeze(),
                "b_H_a": b_H_a.squeeze()}

    @staticmethod
    def _get_target_kpts(kpts_src, h_mtx):
        kpts_trg = torch.matmul(h_mtx, kpts_src.t()).t()
        kpts_trg = [kpt[:2] / kpt[-1] for kpt in kpts_trg]
        kpts_trg = torch.stack(kpts_trg)
        return kpts_trg

    def _get_kpt_fname(self, img_fname):
        kpts_path, img_path = self.kpts_path_p, self.img_path_p
        if 'MegaDepth' in img_fname:
            kpts_path, img_path = self.kpts_path_m, self.img_path_m
        pathname_out = osp.join(kpts_path, img_fname[len(img_path) + 1:])
        pathname_out = pathname_out[:pathname_out.rindex('/')]
        kpt_fname = osp.join(pathname_out, img_fname.split('/')[-1][:-4] + '.pkl')
        return kpt_fname

    def _is_kpt_valid(self, kpt):
        return (0 <= kpt[0] <= self.w_crop - 1) and (0 <= kpt[1] <= self.h_crop - 1)

    def __getitem__(self, item):
        img_fname = self.fnames[item]
        kpt_fname = self._get_kpt_fname(img_fname)

        img = Image.open(img_fname).convert('RGB')

        use_stylization = self.st_path_m is not None and self.st_path_p is not None

        img_st = None
        if use_stylization:
            st_path, img_path = self.st_path_p, self.img_path_p
            if 'MegaDepth' in img_fname:
                st_path, img_path = self.st_path_m, self.img_path_m

            if random.uniform(0, 1) > self.stylized_threshold:
                img_st_fname = osp.join(st_path, img_fname[len(img_path)+1:])

                # double check: if the stylized version does not exist for some reason...
                if not osp.exists(img_st_fname):
                    img_st = img
                else:
                    img_st = Image.open(img_st_fname).convert('RGB')

        crops_data = self._get_random_crops(img, kpt_fname, img_st=img_st)

        kpts_crop_src = crops_data['kpts_crop_src']

        kpts_crop_trg = self._get_target_kpts(kpts_crop_src, crops_data['b_H_a'])
        kpts_crop_src = torch.stack([kpt[:2] for kpt in kpts_crop_src])
        mask_consistency = torch.BoolTensor([self._is_kpt_valid(kpt) for kpt in kpts_crop_trg])

        crop1 = crops_data['crop_src']
        crop2 = crops_data['crop_trg']

        if sum(mask_consistency) == 0:
            kpts_crop_trg = kpts_crop_src.clone()
            crop2 = crop1
        else:
            kpts_crop_src = kpts_crop_src[mask_consistency]
            kpts_crop_trg = kpts_crop_trg[mask_consistency]

        assert kpts_crop_src.shape[0] == kpts_crop_trg.shape[0], 'Number of kpts in src and trg should be equal'

        # Let us pad the number of keypoints up to self.kpts_max to be able to use default collate_fn
        kpts_crop_src_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
        kpts_crop_trg_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
        mask_kpts_collate = torch.BoolTensor(self.n_kpts_crop_max, 1).fill_(0)

        if len(kpts_crop_src) > self.n_kpts_crop_max:
            kpts_crop_src = kpts_crop_src[:self.n_kpts_crop_max]
            kpts_crop_trg = kpts_crop_trg[:self.n_kpts_crop_max]

        kpts_crop_src_pad[:len(kpts_crop_src)] = kpts_crop_src
        kpts_crop_trg_pad[:len(kpts_crop_src)] = kpts_crop_trg
        mask_kpts_collate[:len(kpts_crop_src)] = True

        def kpts_win(kpts, window):
            kpts_out = []
            kpts_r = torch.round(kpts).long()
            for kpt in kpts_r:
                if 0 <= kpt[0] - 1 or kpt[0] + 1 < self.h_crop or 0 <= kpt[1] - 1 or kpt[1] + 1 < self.w_crop:
                    win_kpts = torch.zeros((win.shape[0]**2, 2), dtype=torch.int)
                    win_kpts[:, 0], win_kpts[:, 1] = kpt[0], kpt[1]
                else:
                    win_kpts = kpt.repeat(window.shape[0], 1) + window.to(kpt)
                    win_kpts = list(itertools.product(win_kpts[:, 0].numpy(), win_kpts[:, 1].numpy()))
                    win_kpts = torch.from_numpy(np.array([*win_kpts]))
                kpts_out.append(win_kpts)
            kpts_out = torch.stack(kpts_out).view(-1, 2)
            return kpts_out

        # let us associate each keypoint with a window
        win = torch.arange(-1, 2).unsqueeze(1)
        kpts_src_win = kpts_win(kpts_crop_src_pad, win)
        kpts_trg_win = kpts_win(kpts_crop_trg_pad, win)

        def kpts_win(kpts, window):
            kpts_out = []
            kpts_r = torch.round(kpts).long()
            for kpt in kpts_r:
                if 0 <= kpt[0] - 1 or kpt[0] + 1 < self.h_crop or 0 <= kpt[1] - 1 or kpt[1] + 1 < self.w_crop:
                    win_kpts = torch.zeros((win.shape[0]**2, 2), dtype=torch.int)
                    win_kpts[:, 0], win_kpts[:, 1] = kpt[0], kpt[1]
                else:
                    win_kpts = kpt.repeat(window.shape[0], 1) + window.to(kpt)
                    win_kpts = list(itertools.product(win_kpts[:, 0].numpy(), win_kpts[:, 1].numpy()))
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
        if self.global_desc_dict_m is not None and self.global_desc_dict_p is not None:
            img_fullname = self.fnames[item]
            img_path, kpts_path, dict_knn, fnames_all = \
                self.img_path_p, self.kpts_path_p, self.global_desc_dict_p, self.fnames_p
            if 'MegaDepth' in img_fullname:
                img_path, kpts_path, dict_knn, fnames_all = \
                    self.img_path_m, self.kpts_path_m, self.global_desc_dict_m, self.fnames_m

            img_fname = img_fullname[len(img_path)+1:]

            with open(dict_knn, 'rb') as f:
                hn_fnames_dict = pickle.load(f)

            split = img_path.split('/')[-1]  # train/test

            knn_id = 0
            done = False

            while not done and knn_id < len(hn_fnames_dict[split + '/' + img_fname]):
                img_fname_hn = hn_fnames_dict[split + '/' + img_fname][knn_id]
                img_fname_hn = img_fname_hn[len(split)+1:]
                img_hn = Image.open(osp.join(img_path, img_fname_hn)).convert('RGB')
                img_hn_arr = np.array(img_hn)
                if img_hn_arr.shape[0] > self.h_crop and img_hn_arr.shape[1] > self.w_crop:
                    done = True
                knn_id += 1

            if not done:  # so the image is smaller than the crop
                done_support = False
                # let's take a random image which is larger the crop
                while not done_support:
                    img_id = random.randrange(len(fnames_all))
                    img_fname = fnames_all[img_id]
                    img_fname_hn = img_fname[len(img_path)+1:]
                    img_hn = Image.open(img_fname).convert('RGB')
                    img_hn_arr = np.array(img_hn)
                    if img_hn_arr.shape[0] > self.h_crop and img_hn_arr.shape[1] > self.w_crop:
                        done_support = True

            kpts_hn_fname = osp.join(kpts_path, img_fname_hn[:-4] + '.pkl')
            with open(kpts_hn_fname, 'rb') as f:
                kpts_hn = torch.from_numpy(pickle.load(f))

            # determine crop size
            output_size_a = min(img_hn.size, (self.h_crop, self.w_crop))

            mask_kpts_hn = np.zeros((img_hn_arr.shape[0], img_hn_arr.shape[1]), dtype=np.int)
            val = 1
            for row_id, col_id in zip(kpts_hn[:, 1].int(), kpts_hn[:, 0].int()):
                mask_kpts_hn[row_id, col_id] = val
                val += 1

            mask_kpts_hn_bool = mask_kpts_hn > 0

            def window1(x, size, w):
                l = x - int(0.5 + size / 2)
                r = l + int(0.5 + size)
                if l < 0: l, r = (0, r - l)
                if r > w: l, r = (l + w - r, w)
                if l < 0: l, r = 0, w  # larger than width
                return slice(l, r)

            def window(cx, cy, win_size, scale, img_shape):
                return (window1(cy, win_size[1] * scale, img_shape[0]),
                        window1(cx, win_size[0] * scale, img_shape[1]))

            n_valid_pixel = mask_kpts_hn_bool.sum()
            sample_w = (mask_kpts_hn_bool.astype(np.float) / (1e-16 + n_valid_pixel.astype(np.float))).astype(np.float)

            def sample_valid_pixel():
                p = sample_w.ravel()
                p = p * (1. / (1e-16 + p.sum()))
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
                if trials >= self.n_samples: break  # finished!

                # pick a random valid point from the first image
                if n_valid_pixel == 0: break
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
            #kpts_hn = torch.IntTensor([col_kpts_src, row_kpts_src]).t()

            # let us create padded arrays
            kpts_hn_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
            mask_kpts_hn_collate = torch.BoolTensor(self.n_kpts_crop_max, 1).fill_(0)

            n_valid_kpts = kpts_hn.shape[0]

            if n_valid_kpts > self.n_kpts_crop_max:
                kpts_hn = kpts_hn[:self.n_kpts_crop_max, :]

            kpts_hn_pad[:n_valid_kpts] = kpts_hn
            mask_kpts_hn_collate[:n_valid_kpts] = True

            if self.transforms:
                crop_hn = self.transforms(image=img_a)['image']

        if self.transforms:
            crop1 = self.transforms(image=crop1)['image']
            crop2 = self.transforms(image=crop2)['image']

        return {'img_fname': self.fnames[item],
                'crop_src': crop1,
                'crop_trg': crop2,
                'crop_hn': crop_hn,
                'h_mtx': crops_data['b_H_a'],
                'crop_src_kpts': kpts_crop_src_pad,
                'crop_trg_kpts': kpts_crop_trg_pad,
                'crop_kpts_hn': kpts_hn_pad,
                'kpts_src_win': kpts_src_win,
                'kpts_trg_win': kpts_trg_win,
                'mask_valid_kpts': mask_kpts_collate,
                'mask_valid_kpts_hn': mask_kpts_hn_collate}

    def __len__(self):
        return len(self.fnames)


class AachenSynthHomography(Dataset):
    def __init__(self, img_path, flow_pairs_txt, kpts_path, crop_size=256, win_size=3, st_path=None, transforms=None):
        self.img_path = img_path
        self.flow_pairs_txt = flow_pairs_txt
        self.kpts_path = kpts_path
        self.h_crop, self.w_crop = (crop_size, crop_size)
        self.win_size = win_size
        self.st_path = st_path

        self.stylized_threshold = 0.4
        self.n_kpts_crop_max = 200
        self.transforms = transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Synthetic homography class
        self.homography = HomographyAugmenter(crop_hw=(self.h_crop, self.w_crop))

        self.fnames = []
        with open(self.flow_pairs_txt, 'r') as f:
            for pair_fname in f:
                single_fnames = pair_fname.rstrip()[:-4].split('_')
                fname1, fname2 = single_fnames[0], single_fnames[1]
                self.fnames.append(osp.join(self.img_path, f'{fname1}' + '.jpg'))
                self.fnames.append(osp.join(self.img_path, f'{fname2}' + '.jpg'))

        self.fnames = list(set(self.fnames))

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
        cv_img2crop1, _, h2img, _, crop_center = self.homography.get_random_homography(image_hw=(h, w))
        cv_img2crop2, _, _, h2crop2, _ = self.homography.get_random_homography(image_hw=(h, w),
                                                                               crop_center=crop_center)

        img2 = img_st if img_st is not None else img

        if kpts is not None:
            crop1, kpts1_w = self.homography.warp_image_and_kpts(img, kpts, cv_img2crop1)
            crop2, kpts2_w = self.homography.warp_image_and_kpts(img2, kpts, cv_img2crop2)
        else:
            crop1 = self.homography.warp_image(img, cv_img2crop1)
            crop2 = self.homography.warp_image(img2, cv_img2crop2)
            kpts1_w, kpts2_w = None, None

        a_H_b = torch.from_numpy(np.matmul(h2img, h2crop2)).float().unsqueeze(0)
        b_H_a = torch.inverse(a_H_b)
        return crop1, crop2, kpts1_w, kpts2_w, a_H_b, b_H_a

    def _get_random_crops(self, img, kpts_fname, img_st=None):
        done = False
        n_iter = 0
        stop_search_patch_niter = 10
        std_th = 20

        with open(kpts_fname, 'rb') as f:
            kpts = torch.from_numpy(pickle.load(f))

        if len(kpts.shape) == 3:
            kpts = kpts.squeeze()

        while not done and (n_iter < stop_search_patch_niter):
            crop1, crop2, kpts1, kpts2, a_H_b, b_H_a = self._generate_crops(img, kpts, img_st)
            if len(kpts1) == 0:
                n_iter += 1
                continue

            if crop1.mean(axis=-1).flatten().std() < std_th:
                n_iter += 1
                continue

            a_grid_px_b = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    a_H_b)
            b_grid_px_a = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    b_H_a)

            # normalize coordinates and compute the valid mask
            a_grid_norm_b = K.geometry.conversions.normalize_pixel_coordinates(a_grid_px_b, self.h_crop, self.w_crop)
            b_grid_norm_a = K.geometry.conversions.normalize_pixel_coordinates(b_grid_px_a, self.h_crop, self.w_crop)

            if (not self._is_mask_valid(a_grid_norm_b)) or (not self._is_mask_valid(b_grid_norm_a)):
                n_iter += 1
                continue

            done = True

        if not done:
            crop1, crop2, _, _, _, _ = self._generate_crops(img, None)
            kpts1 = _DetectorBase.get_dummy_kpts(self.h_crop, self.w_crop, self.n_kpts_crop_max).squeeze()
            a_H_b = torch.eye(3).unsqueeze(0)
            b_H_a = a_H_b.clone()
            a_grid_px_b = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    a_H_b)
            b_grid_px_a = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    b_H_a)

        kpts = torch.stack([kpt[:2] for kpt in kpts])

        return {"crop_src": crop1,
                "crop_trg": crop2,
                "kpts_crop_src": kpts1,
                "kpts_img": kpts,
                "crop_size": self.h_crop,
                "a_grid_px_b": a_grid_px_b.squeeze(),
                "b_grid_px_a": b_grid_px_a.squeeze(),
                "a_H_b": a_H_b.squeeze(),
                "b_H_a": b_H_a.squeeze()}

    @staticmethod
    def _get_target_kpts(kpts_src, h_mtx):
        kpts_trg = torch.matmul(h_mtx, kpts_src.t()).t()
        kpts_trg = [kpt[:2] / kpt[-1] for kpt in kpts_trg]
        kpts_trg = torch.stack(kpts_trg)
        return kpts_trg

    def _get_kpt_fname(self, img_fname):
        pathname_out = osp.join(self.kpts_path, img_fname[len(self.img_path) + 1:])
        pathname_out = pathname_out[:pathname_out.rindex('/')]
        kpt_fname = osp.join(pathname_out, img_fname.split('/')[-1][:-4] + '.pkl')
        return kpt_fname

    def _is_kpt_valid(self, kpt):
        return (0 <= kpt[0] <= self.w_crop - 1) and (0 <= kpt[1] <= self.h_crop - 1)

    def __getitem__(self, item):
        img_fname = self.fnames[item]
        kpt_fname = self._get_kpt_fname(img_fname)

        img = Image.open(img_fname).convert('RGB')
        # img_greyscale = Image.open(self.fnames[item]).convert('L')

        img_st = None
        if self.st_path is not None:
            if random.uniform(0, 1) > self.stylized_threshold:
                img_st_fname = osp.join(self.st_path, img_fname[len(self.img_path)+1:])
                # double check: if the stylized version does not exist for some reason...
                if not osp.exists(img_st_fname):
                    img_st = img
                else:
                    img_st = Image.open(img_st_fname).convert('RGB')

        crops_data = self._get_random_crops(img, kpt_fname, img_st=img_st)

        kpts_crop_src = crops_data['kpts_crop_src']

        kpts_crop_trg = self._get_target_kpts(kpts_crop_src, crops_data['b_H_a'])
        kpts_crop_src = torch.stack([kpt[:2] for kpt in kpts_crop_src])
        mask_consistency = torch.BoolTensor([self._is_kpt_valid(kpt) for kpt in kpts_crop_trg])

        crop1 = crops_data['crop_src']
        crop2 = crops_data['crop_trg']

        if sum(mask_consistency) == 0:
            kpts_crop_trg = kpts_crop_src.clone()
            crop2 = crop1
        else:
            kpts_crop_src = kpts_crop_src[mask_consistency]
            kpts_crop_trg = kpts_crop_trg[mask_consistency]

        assert kpts_crop_src.shape[0] == kpts_crop_trg.shape[0], 'Number of kpts in src and trg should be equal'

        # Let us pad the number of keypoints up to self.kpts_max to be able to use default collate_fn
        kpts_crop_src_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
        kpts_crop_trg_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
        mask_kpts_collate = torch.BoolTensor(self.n_kpts_crop_max, 1).fill_(0)

        if len(kpts_crop_src) > self.n_kpts_crop_max:
            kpts_crop_src = kpts_crop_src[:self.n_kpts_crop_max]
            kpts_crop_trg = kpts_crop_trg[:self.n_kpts_crop_max]

        kpts_crop_src_pad[:len(kpts_crop_src)] = kpts_crop_src
        kpts_crop_trg_pad[:len(kpts_crop_src)] = kpts_crop_trg
        mask_kpts_collate[:len(kpts_crop_src)] = True

        def kpts_win(kpts, window):
            kpts_out = []
            kpts_r = torch.round(kpts).long()
            for kpt in kpts_r:
                if 0 <= kpt[0] - 1 or kpt[0] + 1 < self.h_crop or 0 <= kpt[1] - 1 or kpt[1] + 1 < self.w_crop:
                    win_kpts = torch.zeros((win.shape[0]**2, 2), dtype=torch.int)
                    win_kpts[:, 0], win_kpts[:, 1] = kpt[0], kpt[1]
                else:
                    win_kpts = kpt.repeat(window.shape[0], 1) + window.to(kpt)
                    win_kpts = list(itertools.product(win_kpts[:, 0].numpy(), win_kpts[:, 1].numpy()))
                    win_kpts = torch.from_numpy(np.array([*win_kpts]))
                kpts_out.append(win_kpts)
            kpts_out = torch.stack(kpts_out).view(-1, 2)
            return kpts_out

        # let us associate each keypoint with a window
        win = torch.arange(-1, 2).unsqueeze(1)
        kpts_src_win = kpts_win(kpts_crop_src_pad, win)
        kpts_trg_win = kpts_win(kpts_crop_trg_pad, win)

        if self.transforms:
            crop1 = self.transforms(image=crop1)['image']
            crop2 = self.transforms(image=crop2)['image']

        return {'img_fname': self.fnames[item],
                'crop_src': crop1,
                'crop_trg': crop2,
                'h_mtx': crops_data['b_H_a'],
                'crop_src_kpts': kpts_crop_src_pad,
                'crop_trg_kpts': kpts_crop_trg_pad,
                'kpts_src_win': kpts_src_win,
                'kpts_trg_win': kpts_trg_win,
                'mask_valid_kpts': mask_kpts_collate}

    def __len__(self):
        return len(self.fnames)


class AachenFlowDataset(Dataset):
    def __init__(self, img_path, flow_main_path, flow_txt, kpts_path, crop_size=192, transforms=None):
        self.img_path = img_path
        self.optical_flow_path = osp.join(flow_main_path, 'flow')
        self.mask_flow_path = osp.join(flow_main_path, 'mask')
        self.flow_txt = flow_txt
        self.kpts_path = kpts_path
        self.crop_size = crop_size
        self.transforms = transforms

        self.n_samples = 5
        self.n_dummy_kpts = 10
        self.n_kpts_crop_max = 1000

        with open(self.flow_txt, 'r') as f:
            self.pairs_fnames = [fname.rstrip() for fname in f]

    def __getitem__(self, item):
        time_start = time.time()
        pair_fname = self.pairs_fnames[item]
        img1_fname, img2_fname = pair_fname[:-4].split('_')
        img1 = Image.open(osp.join(self.img_path, img1_fname + '.jpg')).convert('RGB')
        img2 = Image.open(osp.join(self.img_path, img2_fname + '.jpg')).convert('RGB')

        mask = np.asarray(Image.open(osp.join(self.mask_flow_path, pair_fname)))
        flow = np.float32(np.asarray(Image.open(osp.join(self.optical_flow_path, pair_fname))).view(np.int16)) / 16
        h, w = flow.shape[:2]
        abs_flow = flow + np.mgrid[:h, :w][::-1].transpose(1, 2, 0)

        kpts1_fname = osp.join(self.kpts_path, img1_fname + '.pkl')
        kpts2_fname = osp.join(self.kpts_path, img2_fname + '.pkl')
        with open(kpts1_fname, 'rb') as f:
            kpts1 = torch.from_numpy(pickle.load(f))
        with open(kpts2_fname, 'rb') as f:
            kpts2 = torch.from_numpy(pickle.load(f))

        mask_kpts = np.zeros(mask.shape, dtype=np.int)
        val = 1
        for row_id, col_id in zip(kpts1[:, 1].int(), kpts1[:, 0].int()):
            mask_kpts[row_id, col_id] = val
            val += 1

        mask_kpts_f = mask_kpts * mask
        mask_kpts_bool = mask_kpts_f > 0
        mask_orig_bool = mask > 0

        '''
        kpts_ids = mask_kpts_f[mask_kpts_f != 0] - 1

        ktps_filtered = torch.zeros([len(kpts_ids), 2], dtype=torch.int32)
        for k, id_ in enumerate(kpts_ids):
            # y, x
            ktps_filtered[k, 0], ktps_filtered[k, 1] = kpts1[id_, 0], kpts1[id_, 1]

        #ktps_filtered[:, 0], ktps_filtered[:, 1] = kpts1[kpts_ids, 0], kpts1[kpts_ids, 1]
        
        print(f'kpts1.shape: {kpts1.shape}, n_kpts: {mask_kpts.sum()}, valid_kpts: {mask_kpts_f.sum()}')
        sys.exit()

        print(f'kpts1[:, 0].max(): {kpts1[:, 0].max()}')
        print(f'kpts1[:, 1].max(): {kpts1[:, 1].max()}')  # kpts1[:, 1] - x-coordinate (rows)
                                                          # kpts1[:, 0] - y-coordinate (cols)
        sys.exit()
        '''

        # determine crop size
        output_size_a = min(img1.size, (self.crop_size, self.crop_size))
        output_size_b = min(img2.size, (self.crop_size, self.crop_size))
        img_a = np.array(img1)
        img_b = np.array(img2)

        ah, aw, p1 = img_a.shape
        bh, bw, p2 = img_b.shape
        assert p1 == 3
        assert p2 == 3
        assert abs_flow.shape == (ah, aw, 2)
        assert mask_kpts_f.shape == (ah, aw)

        # Let's start by computing the scale of the
        # optical flow and applying a median filter:
        dx = np.gradient(abs_flow[:, :, 0])
        dy = np.gradient(abs_flow[:, :, 1])
        scale = np.sqrt(np.clip(np.abs(dx[1] * dy[0] - dx[0] * dy[1]), 1e-16, 1e16))

        accu2 = np.zeros((16, 16), bool)
        Q = lambda x, w: np.int32(16 * (x - w.start) / (w.stop - w.start))

        def window1(x, size, w):
            l = x - int(0.5 + size / 2)
            r = l + int(0.5 + size)
            if l < 0: l, r = (0, r - l)
            if r > w: l, r = (l + w - r, w)
            if l < 0: l, r = 0, w  # larger than width
            return slice(l, r)

        def window(cx, cy, win_size, scale, img_shape):
            return (window1(cy, win_size[1] * scale, img_shape[0]),
                    window1(cx, win_size[0] * scale, img_shape[1]))

        n_valid_pixel = mask_kpts_bool.sum()
        sample_w = (mask_kpts_bool.astype(np.float) / (1e-16 + n_valid_pixel.astype(np.float))).astype(np.float)

        def sample_valid_pixel():
            p = sample_w.ravel()
            p = p * (1. / (1e-16 + p.sum()))

            try:
                n = np.random.choice(sample_w.size, p=p)
                # n = random.choices(range(sample_w.size), weights=p, k=n_elems)
            except:
                n = np.random.choice(sample_w.size)

            y, x = np.unravel_index(n, sample_w.shape)

            return x, y

        #c1x_arr, c1y_arr = sample_valid_pixel(50 * self.n_samples)

        # Find suitable left and right windows
        trials = 0  # take the best out of few trials
        best = -np.inf, None
        for i in range(50 * self.n_samples):
            if trials >= self.n_samples: break  # finished!

            # pick a random valid point from the first image
            if n_valid_pixel == 0: break
            c1x, c1y = sample_valid_pixel() #c1x_arr[i], c1y_arr[i]

            # Find in which position the center of the left
            # window ended up being placed in the right image
            c2x, c2y = (abs_flow[c1y, c1x] + 0.5).astype(np.int32)
            if not (0 <= c2x < bw and 0 <= c2y < bh): continue

            # Get the flow scale
            sigma = scale[c1y, c1x]

            # Determine sampling windows
            if 0.2 < sigma < 1:
                win1 = window(c1x, c1y, output_size_a, 1 / sigma, img_a.shape)
                win2 = window(c2x, c2y, output_size_b, 1, img_b.shape)
            elif 1 <= sigma < 5:
                win1 = window(c1x, c1y, output_size_a, 1, img_a.shape)
                win2 = window(c2x, c2y, output_size_b, sigma, img_b.shape)
            else:
                continue  # bad scale

            # compute a score based on the flow
            x2, y2 = abs_flow[win1].reshape(-1, 2).T.astype(np.int32)
            # Check the proportion of valid flow vectors
            valid = (win2[1].start <= x2) & (x2 < win2[1].stop) \
                    & (win2[0].start <= y2) & (y2 < win2[0].stop)
            score1 = (valid * mask[win1].ravel()).mean()
            # check the coverage of the second window
            accu2[:] = False
            accu2[Q(y2[valid], win2[0]), Q(x2[valid], win2[1])] = True
            score2 = accu2.mean()
            # Check how many hits we got
            score = min(score1, score2)

            trials += 1
            if score > best[0]:
                best = score, win1, win2

        if None in best:  # counldn't find a good windows
            c1x, c1y = sample_valid_pixel() #c1x_arr[0], c1y_arr[0] #(1)
            win1 = window(c1x, c1y, output_size_a, 1, img_a.shape)
            img_a = img_a[win1]
            img_b = img_a.copy()
            mask_kpts_bool = mask_kpts_bool[win1]
            abs_flow = np.nan * np.ones((2,) + output_size_a[::-1], dtype=np.float32)
        else:
            win1, win2 = best[1:]
            img_a = img_a[win1]
            img_b = img_b[win2]
            abs_flow_orig_win = abs_flow[win1] - np.float32([[[win2[1].start, win2[0].start]]])
            mask_kpts_bool = mask_kpts_bool[win1]
            abs_flow_orig_win[~mask_kpts_bool.view(bool)] = np.nan  # mask bad pixels!
            abs_flow = abs_flow_orig_win.transpose(2, 0, 1)  # --> (2,H,W)

            # rescale if necessary
            if img_a.shape[:2][::-1] != output_size_a:
                sx, sy = (np.float32(output_size_a) - 1) / (np.float32(img_a.shape[:2][::-1]) - 1)
                img_a = np.asarray(Image.fromarray(img_a).resize(output_size_a, Image.ANTIALIAS))
                mask_kpts_bool = np.asarray(Image.fromarray(mask_kpts_bool).resize(output_size_a, Image.NEAREST))
                afx = Image.fromarray(abs_flow[0]).resize(output_size_a, Image.NEAREST)
                afy = Image.fromarray(abs_flow[1]).resize(output_size_a, Image.NEAREST)
                abs_flow = np.stack((np.float32(afx), np.float32(afy)))

            if img_b.shape[:2][::-1] != output_size_b:
                sx, sy = (np.float32(output_size_b) - 1) / (np.float32(img_b.shape[:2][::-1]) - 1)
                img_b = np.asarray(Image.fromarray(img_b).resize(output_size_b, Image.ANTIALIAS))
                abs_flow *= [[[sx]], [[sy]]]

        row_kpts_src, col_kpts_src = np.nonzero(mask_kpts_bool)
        col_kpts_trg, row_kpts_trg = (abs_flow[:, row_kpts_src, col_kpts_src] + 0.5).astype(np.int32)

        # Validate transformed keypoints
        valid_kpts = torch.zeros(len(row_kpts_src)).bool()
        for id_, (row_v, col_v) in enumerate(zip(row_kpts_trg, col_kpts_trg)):
            if 0 <= row_v < self.crop_size and 0 <= col_v < self.crop_size:
                valid_kpts[id_] = True

        if valid_kpts.sum() < self.n_dummy_kpts:
            img_b = img_a
            r_src = np.random.randint(0, self.crop_size, self.n_dummy_kpts)
            c_src = np.random.randint(0, self.crop_size, self.n_dummy_kpts)
            kpts_src = torch.IntTensor([r_src, c_src]).t()
            kpts_trg = torch.IntTensor([r_src, c_src]).t()
        else:
            kpts_src = torch.IntTensor([row_kpts_src[valid_kpts], col_kpts_src[valid_kpts]]).t()
            kpts_trg = torch.IntTensor([row_kpts_trg[valid_kpts], col_kpts_trg[valid_kpts]]).t()

        # let us create padded arrays
        kpts_src_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
        kpts_trg_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
        mask_kpts_collate = torch.BoolTensor(self.n_kpts_crop_max, 1).fill_(0)

        n_valid_kpts = kpts_src.shape[0]

        if n_valid_kpts > self.n_kpts_crop_max:
            kpts_src = kpts_src[:self.n_kpts_crop_max, :]
            kpts_trg = kpts_trg[:self.n_kpts_crop_max, :]

        kpts_src_pad[:n_valid_kpts] = kpts_src
        kpts_trg_pad[:n_valid_kpts] = kpts_trg
        mask_kpts_collate[:n_valid_kpts] = True

        if self.transforms:
            img1 = self.transforms(image=img_a)['image']
            img2 = self.transforms(image=img_b)['image']

        return {'crop_src': img1,
                'crop_trg': img2,
                'crop_src_kpts': kpts_src_pad,
                'crop_trg_kpts': kpts_trg_pad,
                'mask_valid_kpts': mask_kpts_collate,
               }

    def __len__(self):
        return len(self.pairs_fnames)


class AachenSHGlobalDescDataset(Dataset):
    def __init__(self,
                 img_path,
                 flow_pairs_txt,
                 kpts_path,
                 crop_size=256,
                 win_size=3,
                 global_desc_dict=None,
                 st_path=None,
                 transforms=None):

        self.img_path = img_path
        self.flow_pairs_txt = flow_pairs_txt
        self.kpts_path = kpts_path
        self.h_crop, self.w_crop = (crop_size, crop_size)
        self.win_size = win_size
        self.global_desc_dict = global_desc_dict
        self.st_path = st_path
        self.n_samples = 5

        self.stylized_threshold = 0.4
        self.n_kpts_crop_max = 200
        self.transforms = transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Synthetic homography class
        self.homography = HomographyAugmenter(crop_hw=(self.h_crop, self.w_crop))

        self.fnames = []
        with open(self.flow_pairs_txt, 'r') as f:
            for pair_fname in f:
                single_fnames = pair_fname.rstrip()[:-4].split('_')
                fname1, fname2 = single_fnames[0], single_fnames[1]
                self.fnames.append(osp.join(self.img_path, f'{fname1}' + '.jpg'))
                self.fnames.append(osp.join(self.img_path, f'{fname2}' + '.jpg'))

        self.fnames = list(set(self.fnames))

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
        cv_img2crop1, _, h2img, _, crop_center = self.homography.get_random_homography(image_hw=(h, w))
        cv_img2crop2, _, _, h2crop2, _ = self.homography.get_random_homography(image_hw=(h, w),
                                                                               crop_center=crop_center)

        img2 = img_st if img_st is not None else img

        if kpts is not None:
            crop1, kpts1_w = self.homography.warp_image_and_kpts(img, kpts, cv_img2crop1)
            crop2, kpts2_w = self.homography.warp_image_and_kpts(img2, kpts, cv_img2crop2)
        else:
            crop1 = self.homography.warp_image(img, cv_img2crop1)
            crop2 = self.homography.warp_image(img2, cv_img2crop2)
            kpts1_w, kpts2_w = None, None

        a_H_b = torch.from_numpy(np.matmul(h2img, h2crop2)).float().unsqueeze(0)
        b_H_a = torch.inverse(a_H_b)
        return crop1, crop2, kpts1_w, kpts2_w, a_H_b, b_H_a

    def _get_random_crops(self, img, kpts_fname, img_st=None):
        done = False
        n_iter = 0
        stop_search_patch_niter = 10
        std_th = 20

        with open(kpts_fname, 'rb') as f:
            kpts = torch.from_numpy(pickle.load(f))

        if len(kpts.shape) == 3:
            kpts = kpts.squeeze()

        while not done and (n_iter < stop_search_patch_niter):
            crop1, crop2, kpts1, kpts2, a_H_b, b_H_a = self._generate_crops(img, kpts, img_st)
            if len(kpts1) == 0:
                n_iter += 1
                continue

            if crop1.mean(axis=-1).flatten().std() < std_th:
                n_iter += 1
                continue

            a_grid_px_b = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    a_H_b)
            b_grid_px_a = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    b_H_a)

            # normalize coordinates and compute the valid mask
            a_grid_norm_b = K.geometry.conversions.normalize_pixel_coordinates(a_grid_px_b, self.h_crop, self.w_crop)
            b_grid_norm_a = K.geometry.conversions.normalize_pixel_coordinates(b_grid_px_a, self.h_crop, self.w_crop)

            if (not self._is_mask_valid(a_grid_norm_b)) or (not self._is_mask_valid(b_grid_norm_a)):
                n_iter += 1
                continue

            done = True

        if not done:
            crop1, crop2, _, _, _, _ = self._generate_crops(img, None)
            kpts1 = _DetectorBase.get_dummy_kpts(self.h_crop, self.w_crop, self.n_kpts_crop_max).squeeze()
            a_H_b = torch.eye(3).unsqueeze(0)
            b_H_a = a_H_b.clone()
            a_grid_px_b = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    a_H_b)
            b_grid_px_a = K.geometry.warp.warp_grid(K.utils.create_meshgrid(self.h_crop, self.w_crop, False),
                                                    b_H_a)

        kpts = torch.stack([kpt[:2] for kpt in kpts])

        return {"crop_src": crop1,
                "crop_trg": crop2,
                "kpts_crop_src": kpts1,
                "kpts_img": kpts,
                "crop_size": self.h_crop,
                "a_grid_px_b": a_grid_px_b.squeeze(),
                "b_grid_px_a": b_grid_px_a.squeeze(),
                "a_H_b": a_H_b.squeeze(),
                "b_H_a": b_H_a.squeeze()}

    @staticmethod
    def _get_target_kpts(kpts_src, h_mtx):
        kpts_trg = torch.matmul(h_mtx, kpts_src.t()).t()
        kpts_trg = [kpt[:2] / kpt[-1] for kpt in kpts_trg]
        kpts_trg = torch.stack(kpts_trg)
        return kpts_trg

    def _get_kpt_fname(self, img_fname):
        pathname_out = osp.join(self.kpts_path, img_fname[len(self.img_path) + 1:])
        pathname_out = pathname_out[:pathname_out.rindex('/')]
        kpt_fname = osp.join(pathname_out, img_fname.split('/')[-1][:-4] + '.pkl')
        return kpt_fname

    def _is_kpt_valid(self, kpt):
        return (0 <= kpt[0] <= self.w_crop - 1) and (0 <= kpt[1] <= self.h_crop - 1)

    def __getitem__(self, item):
        img_fname = self.fnames[item]
        kpt_fname = self._get_kpt_fname(img_fname)

        img = Image.open(img_fname).convert('RGB')
        # img_greyscale = Image.open(self.fnames[item]).convert('L')

        img_st = None
        if self.st_path is not None:
            if random.uniform(0, 1) > self.stylized_threshold:
                img_st_fname = osp.join(self.st_path, img_fname[len(self.img_path)+1:])
                # double check: if the stylized version does not exist for some reason...
                if not osp.exists(img_st_fname):
                    img_st = img
                else:
                    img_st = Image.open(img_st_fname).convert('RGB')

        crops_data = self._get_random_crops(img, kpt_fname, img_st=img_st)

        kpts_crop_src = crops_data['kpts_crop_src']

        kpts_crop_trg = self._get_target_kpts(kpts_crop_src, crops_data['b_H_a'])
        kpts_crop_src = torch.stack([kpt[:2] for kpt in kpts_crop_src])
        mask_consistency = torch.BoolTensor([self._is_kpt_valid(kpt) for kpt in kpts_crop_trg])

        crop1 = crops_data['crop_src']
        crop2 = crops_data['crop_trg']

        if sum(mask_consistency) == 0:
            kpts_crop_trg = kpts_crop_src.clone()
            crop2 = crop1
        else:
            kpts_crop_src = kpts_crop_src[mask_consistency]
            kpts_crop_trg = kpts_crop_trg[mask_consistency]

        assert kpts_crop_src.shape[0] == kpts_crop_trg.shape[0], 'Number of kpts in src and trg should be equal'

        # Let us pad the number of keypoints up to self.kpts_max to be able to use default collate_fn
        kpts_crop_src_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
        kpts_crop_trg_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
        mask_kpts_collate = torch.BoolTensor(self.n_kpts_crop_max, 1).fill_(0)

        if len(kpts_crop_src) > self.n_kpts_crop_max:
            kpts_crop_src = kpts_crop_src[:self.n_kpts_crop_max]
            kpts_crop_trg = kpts_crop_trg[:self.n_kpts_crop_max]

        kpts_crop_src_pad[:len(kpts_crop_src)] = kpts_crop_src
        kpts_crop_trg_pad[:len(kpts_crop_src)] = kpts_crop_trg
        mask_kpts_collate[:len(kpts_crop_src)] = True

        def kpts_win(kpts, window):
            kpts_out = []
            kpts_r = torch.round(kpts).long()
            for kpt in kpts_r:
                if 0 <= kpt[0] - 1 or kpt[0] + 1 < self.h_crop or 0 <= kpt[1] - 1 or kpt[1] + 1 < self.w_crop:
                    win_kpts = torch.zeros((win.shape[0]**2, 2), dtype=torch.int)
                    win_kpts[:, 0], win_kpts[:, 1] = kpt[0], kpt[1]
                else:
                    win_kpts = kpt.repeat(window.shape[0], 1) + window.to(kpt)
                    win_kpts = list(itertools.product(win_kpts[:, 0].numpy(), win_kpts[:, 1].numpy()))
                    win_kpts = torch.from_numpy(np.array([*win_kpts]))
                kpts_out.append(win_kpts)
            kpts_out = torch.stack(kpts_out).view(-1, 2)
            return kpts_out

        # let us associate each keypoint with a window
        win = torch.arange(-1, 2).unsqueeze(1)
        kpts_src_win = kpts_win(kpts_crop_src_pad, win)
        kpts_trg_win = kpts_win(kpts_crop_trg_pad, win)

        crop_hn, kpts_hn = [], []
        # let us find hard negative image based on global representation
        if self.global_desc_dict is not None:
            img_fullname = self.fnames[item]
            img_fname = img_fullname[len(self.img_path)+1:]

            with open(self.global_desc_dict, 'rb') as f:
                hn_fnames_dict = pickle.load(f)

            img_fname_hn = hn_fnames_dict[img_fname][0]
            img_hn = Image.open(osp.join(self.img_path, img_fname_hn)).convert('RGB')
            img_hn_arr = np.array(img_hn)

            kpts_hn_fname = osp.join(self.kpts_path, img_fname_hn[:-4] + '.pkl')
            with open(kpts_hn_fname, 'rb') as f:
                kpts_hn = torch.from_numpy(pickle.load(f))

            # determine crop size
            output_size_a = min(img_hn.size, (self.h_crop, self.w_crop))

            mask_kpts_hn = np.zeros((img_hn_arr.shape[0], img_hn_arr.shape[1]), dtype=np.int)
            val = 1
            for row_id, col_id in zip(kpts_hn[:, 1].int(), kpts_hn[:, 0].int()):
                mask_kpts_hn[row_id, col_id] = val
                val += 1

            mask_kpts_hn_bool = mask_kpts_hn > 0

            def window1(x, size, w):
                l = x - int(0.5 + size / 2)
                r = l + int(0.5 + size)
                if l < 0: l, r = (0, r - l)
                if r > w: l, r = (l + w - r, w)
                if l < 0: l, r = 0, w  # larger than width
                return slice(l, r)

            def window(cx, cy, win_size, scale, img_shape):
                return (window1(cy, win_size[1] * scale, img_shape[0]),
                        window1(cx, win_size[0] * scale, img_shape[1]))

            n_valid_pixel = mask_kpts_hn_bool.sum()
            sample_w = (mask_kpts_hn_bool.astype(np.float) / (1e-16 + n_valid_pixel.astype(np.float))).astype(np.float)

            def sample_valid_pixel():
                p = sample_w.ravel()
                p = p * (1. / (1e-16 + p.sum()))
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
                if trials >= self.n_samples: break  # finished!

                # pick a random valid point from the first image
                if n_valid_pixel == 0: break
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

            # let us create padded arrays
            kpts_hn_pad = torch.FloatTensor(self.n_kpts_crop_max, 2).fill_(0)
            mask_kpts_hn_collate = torch.BoolTensor(self.n_kpts_crop_max, 1).fill_(0)

            n_valid_kpts = kpts_hn.shape[0]

            if n_valid_kpts > self.n_kpts_crop_max:
                kpts_hn = kpts_hn[:self.n_kpts_crop_max, :]

            kpts_hn_pad[:n_valid_kpts] = kpts_hn
            mask_kpts_hn_collate[:n_valid_kpts] = True

            if self.transforms:
                crop_hn = self.transforms(image=img_a)['image']

        if self.transforms:
            crop1 = self.transforms(image=crop1)['image']
            crop2 = self.transforms(image=crop2)['image']

        return {'img_fname': self.fnames[item],
                'crop_src': crop1,
                'crop_trg': crop2,
                'crop_hn': crop_hn,
                'h_mtx': crops_data['b_H_a'],
                'crop_src_kpts': kpts_crop_src_pad,
                'crop_trg_kpts': kpts_crop_trg_pad,
                'crop_kpts_hn': kpts_hn_pad,
                'kpts_src_win': kpts_src_win,
                'kpts_trg_win': kpts_trg_win,
                'mask_valid_kpts': mask_kpts_collate,
                'mask_valid_kpts_hn': mask_kpts_hn_collate,
                }

    def __len__(self):
        return len(self.fnames)


class PrecomputedValidationDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.fnames = [fname for fname in os.listdir(self.path)]

    def __getitem__(self, item):
        fname = self.fnames[item]
        with open(osp.join(self.path, fname), 'rb') as f:
            metadata = pickle.load(f)

        return metadata

    def __len__(self):
        return len(self.fnames)
