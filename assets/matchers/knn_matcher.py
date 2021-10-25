import numpy as np
import cv2
import torch
from assets.matchers.base import BaseMatcher
from experiments.service.utils import desc_similarity


class KnnMatcher(BaseMatcher):
    def __init__(self, matcher_cfg):
        super().__init__(matcher_cfg)
        self.d_threshold = matcher_cfg.matcher_params.feat_distance_threshold

    def match(self, s1: torch.Tensor, s2: torch.Tensor):
        sim = desc_similarity(s1, s2)
        if sim is None:
            self.n_fails += 1
            return np.asarray([]), np.asarray([])

        nn12_dist, nn12 = torch.min(sim, dim=1)
        nn21_dist, nn21 = torch.min(sim, dim=0)

        ids1 = torch.arange(0, sim.shape[0], device=self.device)
        mask = ids1 == nn21[nn12]
        matches_1 = ids1[mask]
        matches_2 = nn12[mask]

        distances = nn12_dist.index_select(dim=0, index=matches_1)
        matches_1_fin = matches_1
        matches_2_fin = matches_2
        if self.d_threshold is not None:
            matches_1_fin = matches_1.masked_select(distances <= self.d_threshold)
            matches_2_fin = matches_2.masked_select(distances <= self.d_threshold)

        matches = torch.stack([matches_1_fin, matches_2_fin])
        distances = nn12_dist.index_select(dim=0, index=ids1[mask])
        return matches.t().data.cpu().numpy().astype(np.uint32)

    def get_inliers_count(self, s1: dict, s2: dict, ransac_threshold=5.0):
        descs_a, kpts_a = s1['descs'], s1['kpts']
        descs_b, kpts_b = s2['descs'], s2['kpts']

        matches = self.match(torch.from_numpy(descs_a).to(self.device),
                             torch.from_numpy(descs_b).to(self.device))
        if len(matches) == 0:
            return 0

        scales_kpts_a = (s1['original_size'] / s1['new_size']).astype(np.float32)
        kpts_a = (kpts_a + .5) * scales_kpts_a[None] - .5

        scales_kpts_b = (s2['original_size'] / s2['new_size']).astype(np.float32)
        kpts_b = (kpts_b + .5) * scales_kpts_b[None] - .5

        pos_a = kpts_a[matches[:, 0], :2]
        pos_b = kpts_b[matches[:, 1], :2]

        _, mask = cv2.findHomography(pos_a, pos_b, cv2.RANSAC, ransac_threshold)
        n_inliers = sum(mask.ravel().tolist())
        return n_inliers, pos_a, pos_b
