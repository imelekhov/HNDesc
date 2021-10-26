import numpy as np
from PIL import Image
import itertools
import cv2
import torch
import torch.nn as nn


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor
    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629
    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
        'max_keypoints_480x640': None,
        'resize_max': None
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        self.load_state_dict(torch.load(self.config["snapshot"]))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    @staticmethod
    def simple_nms(scores, nms_radius: int):
        """ Fast Non-maximum suppression to remove nearby points """
        assert (nms_radius >= 0)

        def max_pool(x):
            return torch.nn.functional.max_pool2d(
                x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

        zeros = torch.zeros_like(scores)
        max_mask = scores == max_pool(scores)
        for _ in range(2):
            supp_mask = max_pool(max_mask.float()) > 0
            supp_scores = torch.where(supp_mask, zeros, scores)
            new_max_mask = supp_scores == max_pool(supp_scores)
            max_mask = max_mask | (new_max_mask & (~supp_mask))
        return torch.where(max_mask, scores, zeros)

    @staticmethod
    def remove_borders(keypoints, scores, border: int, height: int, width: int):
        """ Removes keypoints too close to the border """
        mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
        mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
        mask = mask_h & mask_w
        return keypoints[mask], scores[mask]

    @staticmethod
    def top_k_keypoints(keypoints, scores, k: int):
        if k >= len(keypoints):
            return keypoints, scores
        scores, indices = torch.topk(scores, k, dim=0)
        return keypoints[indices], scores

    @staticmethod
    def sample_descriptors_orig(keypoints, descriptors, swap_xy: bool = False, s: int = 8):
        """ Interpolate descriptors at keypoint locations """
        b, c, h, w = descriptors.shape
        keypoints = keypoints - s / 2 + 0.5
        keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                                  ).to(keypoints)[None]
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)

        if swap_xy:
            kpts_tmp = keypoints.clone()
            keypoints[:, :, 0], keypoints[:, :, 1] = kpts_tmp[:, :, 1], kpts_tmp[:, :, 0]

        args = {'align_corners': True} if int(torch.__version__[2]) > 2 else {}
        descriptors = torch.nn.functional.grid_sample(
            descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
        descriptors = torch.nn.functional.normalize(
            descriptors.reshape(b, c, -1), p=2, dim=1)
        return descriptors

    @staticmethod
    def sample_descriptors(keypoints, descriptors, s: int = 8):
        """ Interpolate descriptors at keypoint locations """
        b, c, h, w = descriptors.shape
        _, n_kpts, _ = keypoints.shape

        keypoints_r = torch.round(keypoints).long()

        descs_pos = []
        kpts_pos = []
        win = torch.arange(-1, 2).unsqueeze(1)
        for descs, kpt_arr in zip(descriptors, keypoints_r):
            descs_per_batch = []
            kpts_per_batch = []
            for kpt in kpt_arr:
                kpt = kpt.cpu().numpy()
                kpt[0], kpt[1] = kpt[1], kpt[0]
                if 0 <= kpt[0] - 1 or kpt[0] + 1 < h or 0 <= kpt[1] - 1 or kpt[1] + 1 < w:
                    win_kpts = torch.zeros((win.shape[0] ** 2, 2), dtype=torch.int)
                    win_kpts[:, 0], win_kpts[:, 1] = kpt[0], kpt[1]
                else:
                    win_kpts = kpt.repeat(win.shape[0], 1) + win.to(kpt)
                    win_kpts = list(itertools.product(win_kpts[:, 0].cpu().numpy(), win_kpts[:, 1].cpu().numpy()))
                    win_kpts = torch.from_numpy(np.array([*win_kpts]))
                kpts_per_batch.append(win_kpts)
                desc_per_window = descs[:, win_kpts[:, 0].long(), win_kpts[:, 1].long()].mean(dim=-1)
                descs_per_batch.append(desc_per_window)
            kpts_pos.append(torch.stack(kpts_per_batch).view(n_kpts * 9, 2))
            # descs_pos.append(torch.stack(descs_per_batch).view(c, -1))
            descs_pos.append(torch.stack(descs_per_batch).permute(1, 0))
        # descs_pos = torch.stack(descs_pos).view(b, c, -1)
        descs_pos = torch.stack(descs_pos)
        kpts_pos = torch.stack(kpts_pos).view(b, n_kpts * 9, 2)
        descs_pos = torch.nn.functional.normalize(descs_pos,
                                                  p=2,
                                                  dim=1)
        return descs_pos, kpts_pos

    @staticmethod
    def sample_descriptors_batch(kpts, descs, s=8):
        descriptors, kpts = SuperPoint.sample_descriptors(kpts, descs, s)
        # [b, n_kpts, c]
        return descriptors.permute(0, 2, 1), kpts

    @staticmethod
    def sample_descriptors_window_batch(kpts, descs, window):
        _, n_kpts, _ = kpts.shape
        w_size = window * window
        n_kpts_per_w = int(n_kpts / w_size)
        descs_f = torch.cat(
            [desc_w[:, kpt_w[:, 1].long(), kpt_w[:, 0].long()].reshape(-1, n_kpts_per_w, w_size).mean(dim=-1).unsqueeze(0) for
             kpt_w, desc_w in zip(kpts, descs)])
        descs_f = torch.nn.functional.normalize(descs_f,
                                                p=2,
                                                dim=1)

        # [b, n_kpts, c]
        return descs_f.permute(0, 2, 1)

    @staticmethod
    def sample_descriptors_int_batch(kpts, descs, reverse=True):
        _, n_kpts, _ = kpts.shape
        if reverse:
            descs_f = torch.cat(
                [desc_w[:, kpt_w[:, 1].long(), kpt_w[:, 0].long()].unsqueeze(0) for
                 kpt_w, desc_w in zip(kpts, descs)])
        else:
            descs_f = torch.cat(
                [desc_w[:, kpt_w[:, 0].long(), kpt_w[:, 1].long()].unsqueeze(0) for
                 kpt_w, desc_w in zip(kpts, descs)])

        descs_f = torch.nn.functional.normalize(descs_f,
                                                p=2,
                                                dim=1).permute(0, 2, 1)  # [b, n_kpts, desc_length]

        # [b, n_kpts, c]
        return descs_f

    @staticmethod
    def sample_descriptors_round_batch(kpts, descs, reverse=True):
        _, n_kpts, _ = kpts.shape
        if reverse:
            descs_f = torch.cat(
                [desc_w[:, kpt_w[:, 1].long(), kpt_w[:, 0].long()].unsqueeze(0) for
                 kpt_w, desc_w in zip(kpts, descs)])
        else:
            descs_f = torch.cat(
                [desc_w[:, kpt_w[:, 0].long(), kpt_w[:, 1].long()].unsqueeze(0) for
                 kpt_w, desc_w in zip(kpts, descs)])

        descs_f = torch.nn.functional.normalize(descs_f,
                                                p=2,
                                                dim=1)

        # [b, c, n_kpts]
        return descs_f

    def get_resize_max_img(self):
        return self.config['resize_max']

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

        net_input = torch.from_numpy(img_frame / 255.).float()[None, None].to(device)

        data = {'net_input': net_input,
                'original_size': np.array(size),
                'new_size': np.array(size_new)}
        return data

    @staticmethod
    def rescale_kpts(rescaled_kpts, rescaled_size, original_size):
        scales = (original_size / rescaled_size).astype(np.float32)
        kpts = (rescaled_kpts + .5) * scales[None] - .5
        return kpts

    def forward(self, img):
        """ Compute keypoints, scores, descriptors for image """

        # Shared Encoder
        x = self.relu(self.conv1a(img))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = self.simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            self.remove_borders(k, s, self.config['remove_borders'], h * 8, w * 8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints_480x640'] is not None:
            max_n_kpts = self.config['max_keypoints_480x640']
        elif self.config['max_keypoints'] >= 0:
            max_n_kpts = self.config['max_keypoints']
        else:
            max_n_kpts = None

        if max_n_kpts is not None:
            keypoints, scores = list(zip(*[
                self.top_k_keypoints(k, s, max_n_kpts)
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        dense_descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [self.sample_descriptors_orig(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, dense_descriptors)]

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
            'dense_descriptors': dense_descriptors
        }
