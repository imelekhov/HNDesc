from abc import ABC

import random
import torch
import torch.nn as nn
from assets.loss import APLoss


class APCriterion(nn.Module, ABC):
    def __init__(self, knn=20, nq=20, min_val=0, max_val=1, euc=False):
        super().__init__()
        self.knn = knn
        self.ap_criterion_r2d2 = APLoss(nq, min_val, max_val, euc)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, anc_feat, pos_feat, kpts_crop_ids):
        anc_feat_f, pos_feat_f = [], []
        n = 0
        kpts_crop_ids_only_negs = []
        # let us filter out image pairs with small amount of keypoints
        for kpts_per_crop in kpts_crop_ids:
            # if there is only one keypoint - skip this image
            if kpts_per_crop > 1:
                anc_feat_f.append(anc_feat[n:n + kpts_per_crop, :])
                pos_feat_f.append(pos_feat[n:n + kpts_per_crop, :])
                kpts_crop_ids_only_negs.append(kpts_per_crop - 1)
            n += kpts_per_crop

        anc_feat_f, pos_feat_f = torch.cat(anc_feat_f), torch.cat(pos_feat_f)

        b, _ = anc_feat_f.shape

        # b x b matrix
        lbl_pos = torch.eye(b, dtype=torch.bool).to(self.device)
        # compute the similarity matrix
        sim_mtx = torch.mm(anc_feat_f, pos_feat_f.t())
        # similarity for positives [b x 1]
        sim_pos = sim_mtx[lbl_pos].unsqueeze(-1)
        # similarity for all negatives [b x b-1]
        sim_neg_all = sim_mtx[~lbl_pos].view(b, -1)

        # let us sample only self.knn hard negative keypoints from each image crop
        sim_neg = []
        r, c = 0, 0

        n_crops = 0
        for kpts_neg_per_crop in kpts_crop_ids_only_negs:
            kpts_per_crop = kpts_neg_per_crop + 1

            # First, let us keep the similarity values ONLY for keypoints which belong to the remaining images
            # in a mini-batch
            if c == 0 and c + kpts_neg_per_crop < sim_neg_all.shape[1]:
                sim_neg_excl = sim_neg_all[:, c + kpts_neg_per_crop:]
            elif c != 0 and c + kpts_neg_per_crop == sim_neg_all.shape[1]:
                sim_neg_excl = sim_neg_all[:, :c]
            else:
                sim_neg_excl = torch.cat((sim_neg_all[:, :c], sim_neg_all[:, c + kpts_neg_per_crop:]), dim=1)

            neg_top_sim, _ = torch.sort(sim_neg_excl[r:r + kpts_per_crop, :], dim=1, descending=True)

            r += kpts_per_crop
            c += kpts_per_crop

            if neg_top_sim.shape[1] < self.knn:
                # if the number of hard-negatives is less than self.knn, let's repeat the last column
                # (self.knn - number_of_negatives_in_a_minibatch)-times
                n_times = self.knn - neg_top_sim.shape[1]
                neg_pad = torch.cat(n_times * [neg_top_sim[:, -1]]).view(n_times, -1).t()
                neg_top_sim = torch.cat((neg_top_sim, neg_pad), dim=1)
                sim_neg.append(neg_top_sim)
            else:
                sim_neg.append(neg_top_sim[:, :self.knn])

            n_crops += 1

        sim_neg = torch.cat(sim_neg)

        # concatenate similarities for positives and negatives
        sim_final = torch.cat((sim_pos, sim_neg), 1)
        # generate gt labels
        lbl_final = torch.zeros_like(sim_final)
        lbl_final[:, 0] = 1

        # compute AP
        ap = self.ap_criterion_r2d2.compute_AP(sim_final, lbl_final)
        return (1 - ap).mean(), ap.mean()


class APCriterionAllNegs(nn.Module, ABC):
    def __init__(self, nq=20, min_val=0, max_val=1, euc=False):
        super().__init__()
        self.ap_criterion_r2d2 = APLoss(nq, min_val, max_val, euc)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, anc_feat, pos_feat, kpts_crop_ids):
        anc_feat_f, pos_feat_f = [], []
        n = 0
        kpts_crop_ids_only_negs = []
        # let us filter out image pairs with small amount of keypoints
        for kpts_per_crop in kpts_crop_ids:
            # if there is only one keypoint - skip this image
            if kpts_per_crop > 1:
                anc_feat_f.append(anc_feat[n:n + kpts_per_crop, :])
                pos_feat_f.append(pos_feat[n:n + kpts_per_crop, :])
                kpts_crop_ids_only_negs.append(kpts_per_crop - 1)
            n += kpts_per_crop

        anc_feat_f, pos_feat_f = torch.cat(anc_feat_f), torch.cat(pos_feat_f)

        b, _ = anc_feat_f.shape

        # b x b matrix
        lbl_pos = torch.eye(b, dtype=torch.bool).to(self.device)
        # compute the similarity matrix
        sim_mtx = torch.mm(anc_feat_f, pos_feat_f.t())
        # similarity for positives [b x 1]
        sim_pos = sim_mtx[lbl_pos].unsqueeze(-1)
        # similarity for all negatives [b x b-1]
        sim_neg_all = sim_mtx[~lbl_pos].view(b, -1)

        # concatenate similarities for positives and negatives
        sim_final = torch.cat((sim_pos, sim_neg_all), 1)

        # generate gt labels
        lbl_final = torch.zeros_like(sim_final)
        lbl_final[:, 0] = 1

        # compute AP
        ap = self.ap_criterion_r2d2.compute_AP(sim_final, lbl_final)
        return (1 - ap).mean(), ap.mean()


class APCriterionRndNegs(nn.Module, ABC):
    def __init__(self, knn=20, nq=20, min_val=0, max_val=1, euc=False):
        super().__init__()
        self.knn = knn
        self.ap_criterion_r2d2 = APLoss(nq, min_val, max_val, euc)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, anc_feat, pos_feat, kpts_crop_ids):
        anc_feat_f, pos_feat_f = [], []
        n = 0
        kpts_crop_ids_only_negs = []
        # let us filter out image pairs with small amount of keypoints
        for kpts_per_crop in kpts_crop_ids:
            # if there is only one keypoint - skip this image
            if kpts_per_crop > 1:
                anc_feat_f.append(anc_feat[n:n + kpts_per_crop, :])
                pos_feat_f.append(pos_feat[n:n + kpts_per_crop, :])
                kpts_crop_ids_only_negs.append(kpts_per_crop - 1)
            n += kpts_per_crop

        anc_feat_f, pos_feat_f = torch.cat(anc_feat_f), torch.cat(pos_feat_f)

        b, _ = anc_feat_f.shape

        # b x b matrix
        lbl_pos = torch.eye(b, dtype=torch.bool).to(self.device)
        # compute the similarity matrix
        sim_mtx = torch.mm(anc_feat_f, pos_feat_f.t())
        # similarity for positives [b x 1]
        sim_pos = sim_mtx[lbl_pos].unsqueeze(-1)
        # similarity for all negatives [b x b-1]
        sim_neg_all = sim_mtx[~lbl_pos].view(b, -1)

        # let us sample only self.knn hard negative keypoints from each image crop
        sim_neg = []
        r, c = 0, 0

        n_crops = 0
        for kpts_neg_per_crop in kpts_crop_ids_only_negs:
            kpts_per_crop = kpts_neg_per_crop + 1

            # First, let us keep the similarity values ONLY for keypoints which belong to the remaining images
            # in a mini-batch
            if c == 0 and c + kpts_neg_per_crop < sim_neg_all.shape[1]:
                sim_neg_excl = sim_neg_all[:, c + kpts_neg_per_crop:]
            elif c != 0 and c + kpts_neg_per_crop == sim_neg_all.shape[1]:
                sim_neg_excl = sim_neg_all[:, :c]
            else:
                sim_neg_excl = torch.cat((sim_neg_all[:, :c], sim_neg_all[:, c + kpts_neg_per_crop:]), dim=1)

            neg_idxs = list(range(sim_neg_excl.shape[1]))
            random.shuffle(neg_idxs)
            neg_idxs = neg_idxs[:self.knn]
            sim_neg.append(sim_neg_excl[:, neg_idxs])

            r += kpts_per_crop
            c += kpts_per_crop

            n_crops += 1

        sim_neg = torch.cat(sim_neg, dim=1)

        # concatenate similarities for positives and negatives
        sim_final = torch.cat((sim_pos, sim_neg), 1)
        # generate gt labels
        lbl_final = torch.zeros_like(sim_final)
        lbl_final[:, 0] = 1

        # compute AP
        ap = self.ap_criterion_r2d2.compute_AP(sim_final, lbl_final)
        return (1 - ap).mean(), ap.mean()


class APCriterionWithinPair(APCriterion, ABC):
    def __init__(self, knn=20, nq=20, min_val=0, max_val=1, euc=False):
        super().__init__(knn, nq, min_val, max_val, euc)

    def forward(self, anc_feat, pos_feat, kpts_crop_ids):
        anc_feat_f, pos_feat_f = [], []
        n = 0
        kpts_crop_ids_only_negs = []
        # let us filter out image pairs with small amount of keypoints
        for kpts_per_crop in kpts_crop_ids:
            # if there is only one keypoint - skip this image
            if kpts_per_crop > 1:
                anc_feat_f.append(anc_feat[n:n+kpts_per_crop, :])
                pos_feat_f.append(pos_feat[n:n+kpts_per_crop, :])
                kpts_crop_ids_only_negs.append(kpts_per_crop - 1)
            n += kpts_per_crop

        anc_feat_f, pos_feat_f = torch.cat(anc_feat_f), torch.cat(pos_feat_f)

        b, _ = anc_feat_f.shape

        # b x b matrix
        lbl_pos = torch.eye(b, dtype=torch.bool).to(self.device)
        # compute the similarity matrix
        sim_mtx = torch.mm(anc_feat_f, pos_feat_f.t())
        # similarity for positives [b x 1]
        sim_pos = sim_mtx[lbl_pos].unsqueeze(-1)
        # similarity for all negatives [b x b-1]
        sim_neg_all = sim_mtx[~lbl_pos].view(b, -1)

        sim_neg = []
        r, c = 0, 0
        n_crops = 0
        for kpts_neg_per_crop in kpts_crop_ids_only_negs:
            kpts_per_crop = kpts_neg_per_crop + 1
            neg_top_sim, _ = torch.sort(sim_neg_all[r:r + kpts_per_crop, c:c + kpts_neg_per_crop],
                                        dim=1,
                                        descending=True)
            r += kpts_per_crop
            c += kpts_per_crop

            if len(neg_top_sim.shape) == 1:
                neg_top_sim = neg_top_sim.unsqueeze(0)

            if kpts_neg_per_crop < self.knn:
                # if the number of hard-negatives is less than self.knn, let's repeat the last column
                # (self.knn - kpts_neg_per_crop)-times
                n_times = self.knn - kpts_neg_per_crop
                neg_pad = torch.cat(n_times * [neg_top_sim[:, -1]]).view(n_times, -1).t()
                neg_top_sim = torch.cat((neg_top_sim, neg_pad), dim=1)
                sim_neg.append(neg_top_sim)
            else:
                sim_neg.append(neg_top_sim[:, :self.knn])

            n_crops += 1

        sim_neg = torch.cat(sim_neg)
        # concatenate similarities for positives and negatives
        sim_final = torch.cat((sim_pos, sim_neg), 1)

        # generate gt labels
        lbl_final = torch.zeros_like(sim_final)
        lbl_final[:, 0] = 1

        # compute AP
        ap = self.ap_criterion_r2d2.compute_AP(sim_final, lbl_final)
        return (1 - ap).mean(), ap.mean()


class APCriterionInBatchGD(nn.Module, ABC):
    def __init__(self, knn=20, nq=20, min_val=0, max_val=1, euc=False):
        super().__init__()
        self.knn = knn
        self.ap_criterion_r2d2 = APLoss(nq, min_val, max_val, euc)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, anc_feat, pos_feat, hn_feat, kpts_crop_ids):
        b, _ = anc_feat.shape
        pos_similarity = -1e5

        feat_all = torch.cat((pos_feat, hn_feat), dim=0)

        # compute the similarity matrix
        sim_mtx = torch.mm(anc_feat, feat_all.t())

        # similarity for positives [b x 1]
        sim_pos = sim_mtx.diag().unsqueeze(-1)

        sim_mtx.fill_diagonal_(pos_similarity)

        sim_neg = []
        r, c = 0, 0
        for n_kpts_per_crop in kpts_crop_ids:
            # First, let us keep the similarity values ONLY for keypoints which belong to the remaining images
            # in a mini-batch
            if c == 0 and c + n_kpts_per_crop < sim_mtx.shape[1]:
                sim_remain = sim_mtx[r:r+n_kpts_per_crop, c+n_kpts_per_crop:]
            elif c != 0 and c + n_kpts_per_crop == sim_mtx.shape[1]:
                sim_remain = sim_mtx[r:r+n_kpts_per_crop, :c]
            else:
                sim_remain = torch.cat((sim_mtx[r:r+n_kpts_per_crop, :c],
                                        sim_mtx[r:r+n_kpts_per_crop, c+n_kpts_per_crop:]), dim=1)

            neg_top_sim, _ = torch.topk(sim_remain, k=self.knn, dim=1, largest=True)
            sim_neg.append(neg_top_sim)

            r += n_kpts_per_crop
            c += n_kpts_per_crop

        sim_neg = torch.cat(sim_neg)

        # concatenate similarities for positives and negatives
        sim_final = torch.cat((sim_pos, sim_neg), 1)

        # generate gt labels
        lbl_final = torch.zeros_like(sim_final)
        lbl_final[:, 0] = 1

        # compute AP
        ap = self.ap_criterion_r2d2.compute_AP(sim_final, lbl_final)
        return (1 - ap).mean(), ap.mean()


class TripletCriterion(nn.Module, ABC):
    def __init__(self, margin=1, swap=False, hn_type='minibatch'):
        super().__init__()
        assert hn_type in ['minibatch', 'pair']
        self.margin = margin
        self.swap = swap
        self.knn = 1
        self.hn_type = hn_type
        self.criterion = nn.TripletMarginLoss(margin=self.margin, swap=self.swap)

    def forward(self, anc_feat, pos_feat, kpts_crop_ids):

        n, d = anc_feat.shape
        pos_dist = 1e5

        # Let us compute the distance matrix
        anc_feat_mtx = anc_feat.unsqueeze(1).expand(n, n, d)
        pos_feat_mtx = pos_feat.unsqueeze(0).expand(n, n, d)
        dist_mtx = torch.pow(anc_feat_mtx - pos_feat_mtx, 2).sum(2)
        # since the diagonal elements are for positives, let us put a large number there
        dist_mtx.fill_diagonal_(pos_dist)

        if self.hn_type == 'batch':
            # let us find the hard negative for each row (keypoint) by finding an ID with the smallest distance
            _, hn_indices = torch.topk(dist_mtx, k=self.knn, dim=1, largest=False)

            # get the feature vectors for hard negatives
            neg_feat = pos_feat[hn_indices.squeeze()]
        else:
            neg_feat = []
            r, c = 0, 0
            for n_kpts_per_crop in kpts_crop_ids:
                dist_mtx_per_crop = dist_mtx[r:r + n_kpts_per_crop, c:c + n_kpts_per_crop]
                _, hn_indices_per_crop = torch.topk(dist_mtx_per_crop, k=self.knn, dim=1, largest=False)
                pos_feat_per_crop = pos_feat[r:r + n_kpts_per_crop, :]
                neg_feat_per_crop = pos_feat_per_crop[hn_indices_per_crop.squeeze()]
                neg_feat.append(neg_feat_per_crop)
                r += n_kpts_per_crop
                c += n_kpts_per_crop
            neg_feat = torch.cat(neg_feat)

        # compute TripletLoss
        out = self.criterion(anchor=anc_feat, positive=pos_feat, negative=neg_feat)
        return out


class APCriterionWeighted(nn.Module, ABC):
    def __init__(self, knn=20, nq=20, min=0, max=1, euc=False):
        super().__init__()
        self.knn = knn
        self.ap_criterion_r2d2 = APLoss(nq, min, max, euc)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, anc_feat, pos_feat, kpts_crop_ids):
        b, _ = anc_feat.shape

        kpts_crop_ids_only_negs = kpts_crop_ids - 1

        lbl_pos = torch.eye(b, dtype=torch.bool).to(self.device)

        sim_mtx = torch.mm(anc_feat, pos_feat.t())
        sim_self_mtx = torch.mm(pos_feat, pos_feat.t())
        sim_mtx = sim_mtx * 1/(sim_mtx.detach() * sim_self_mtx.detach())  # weight anc sim with pos self sim

        # similarity for positives
        sim_pos = sim_mtx[lbl_pos].unsqueeze(-1)
        # similarity for all negatives
        sim_neg_all = sim_mtx[~lbl_pos].view(b, -1)

        # let us sample only self.knn hard negative keypoints from each image crop
        sim_neg = []
        k = 0
        n_crops = 0
        for kpts_neg_per_crop in kpts_crop_ids_only_negs:
            if kpts_neg_per_crop < 0:
                continue

            if kpts_neg_per_crop < self.knn:
                k += kpts_neg_per_crop
                continue
            neg_top_sim, _ = torch.sort(sim_neg_all[:, k:k + kpts_neg_per_crop], dim=1, descending=True)
            sim_neg.append(neg_top_sim[:, :self.knn])
            k += kpts_neg_per_crop
            n_crops += 1
        sim_neg = torch.stack(sim_neg, dim=1).view(b, self.knn * n_crops)

        # concatenate similarities for positives and negatives
        sim_final = torch.cat((sim_pos, sim_neg), 1)
        # generate gt labels
        lbl_final = torch.zeros_like(sim_final)
        lbl_final[:, 0] = 1

        '''
        # create a label matrix
        lbl_final = torch.eye(b, dtype=torch.float).repeat(2, 2).to(anc_feat)

        # Adjacency matrix
        all_desc = torch.cat((anc_feat, pos_feat), dim=0)
        sim_final = torch.mm(all_desc, all_desc.t())
        '''
        '''
        # create a label matrix
        lbl_final = torch.eye(b, dtype=torch.float).to(anc_feat)

        # Adjacency matrix
        sim_final = torch.mm(anc_feat, pos_feat.t())
        '''

        # compute AP
        ap = self.ap_criterion_r2d2.compute_AP(sim_final, lbl_final)
        return (1 - ap).mean(), ap.mean()
