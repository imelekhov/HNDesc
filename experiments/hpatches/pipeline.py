import os
from os import path as osp
import numpy as np
from tqdm import tqdm
import pickle
import cv2
import torch
from experiments.service.benchmark_base import Benchmark
from experiments.service.ldd_factory import LocalDetectorDescriptor
from experiments.service.matchers_factory import MatchersFactory
from experiments.service.utils import compute_homography_error


def warp_keypoints(keypoints, H):
    """Warp keypoints given a homography
    Parameters
    ----------
    keypoints: numpy.ndarray (N,2)
        Keypoint vector.
    H: numpy.ndarray (3,3)
        Homography.
    Returns
    -------
    warped_keypoints: numpy.ndarray (N,2)
        Warped keypoints vector.
    """
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H))
    return warped_points[:, :2] / warped_points[:, 2:]


def scale_homography(homography, original_scale, new_scale, pre):
    scales = np.divide(new_scale, original_scale)
    if pre:
        s = np.diag(np.append(scales, 1.))
        homography = np.matmul(s, homography)
    else:
        sinv = np.diag(np.append(1. / scales, 1.))
        homography = np.matmul(homography, sinv)
    return homography


class HPSequenceBenchmark(Benchmark):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.hpsequences_path = self.cfg.task.task_params.paths.img_path
        self.preds = self.cfg.task.task_params.output.precomputed_feats_dir

        # Scenes with a very large image resolution (need to be removed)
        self.outliers = ['i_contruction', 'i_crownnight', 'i_dc', 'i_pencils', 'i_whitebuilding',
                         'v_artisans', 'v_astronautis', 'v_talent']

        # Matcher
        self.matcher = MatchersFactory(cfg.matcher).get_matcher()

        # Local detector-descriptor
        self.ldd_model = LocalDetectorDescriptor(self.cfg)

        self.kpts_matches = {}

        self.stats_feat_matches = {"i": {"feat": [], "matches": []},
                                   "v": {"feat": [], "matches": []}}

        self.n_illum = 0
        self.n_viewpnt = 0

        self.h_mtx_ransac_params = {"thr_px": 3.0,
                                    "max_iters": 5000,
                                    "confidence": 0.9995,
                                    }

    def _get_kpts_and_matches(self):
        print("Let us find matches...")
        for seq_name in sorted(os.listdir(self.preds)):
            if seq_name in set(self.outliers):
                continue

            if seq_name[0] == "i":
                self.n_illum += 1
            else:
                self.n_viewpnt += 1

            with open(osp.join(self.preds, seq_name, '1.pkl'), 'rb') as f:
                data = pickle.load(f)
            keypoints_a, descriptor_a = data['kpts'], data['descs']

            for img_id in range(2, 7):
                with open(osp.join(self.preds, seq_name, f'{img_id}.pkl'), 'rb') as f:
                    data = pickle.load(f)
                keypoints_b, descriptor_b = data['kpts'], data['descs']

                self.stats_feat_matches[seq_name[0]]["feat"].append(keypoints_a.shape[0])
                self.stats_feat_matches[seq_name[0]]["feat"].append(keypoints_b.shape[0])

                matches = self.matcher.match(torch.from_numpy(descriptor_a).to(self.device),
                                             torch.from_numpy(descriptor_b).to(self.device))

                self.stats_feat_matches[seq_name[0]]["matches"].append(matches.shape[0])

                pos_a = keypoints_a[matches[:, 0], :2]
                pos_b = keypoints_b[matches[:, 1], :2]

                self.kpts_matches[seq_name + "1_" + str(img_id)] = {"pos_a": pos_a,
                                                                    "pos_b": pos_b}

        print("Let us find matches... Done!")
        return True

    def h_mtx_estimation_benchmark(self, h_thresh):
        h_mtx_res = {"i": [], "v": []}

        for seq_name in sorted(os.listdir(self.hpsequences_path)):
            if seq_name in set(self.outliers):
                continue

            for img_id in range(2, 7):
                img = cv2.imread(os.path.join(self.hpsequences_path, seq_name, str(img_id) + ".ppm"), -1)
                h, w, _ = img.shape

                h_gt = np.loadtxt(os.path.join(self.hpsequences_path, seq_name, "H_1_" + str(img_id)))
                pos_a = self.kpts_matches[seq_name + "1_" + str(img_id)]["pos_a"]
                pos_b = self.kpts_matches[seq_name + "1_" + str(img_id)]["pos_b"]

                h_est, _ = cv2.findHomography(pos_a,
                                              pos_b,
                                              cv2.RANSAC,
                                              self.h_mtx_ransac_params["thr_px"],
                                              maxIters=self.h_mtx_ransac_params["max_iters"],
                                              confidence=self.h_mtx_ransac_params["confidence"])
                if h_est is None:
                    print("No homography found! Sequence name: ", seq_name)
                    sys.exit()

                error_h = compute_homography_error(h_est, h_gt, h, w)
                correct = ((error_h < h_thresh) if error_h is not None else False)
                h_mtx_res[seq_name[0]].append(correct)

        return h_mtx_res

    def _keep_shared_points(self, keypoints, descriptors, H, shape):
        """
        Compute a list of keypoints from the map, filter the list of points by keeping
        only the points that once mapped by H are still inside the shape of the map
        and keep at most 'keep_k_points' keypoints in the image.

        Parameters
        ----------
        keypoints: numpy.ndarray (N,3)
            Keypoint vector, consisting of (x,y,probability).
        descriptors: numpy.ndarray (N,256)
            Keypoint descriptors.
        H: numpy.ndarray (3,3)
            Homography.
        shape: tuple
            Image shape.
        keep_k_points: int
            Number of keypoints to select, based on probability.
        Returns
        -------
        selected_points: numpy.ndarray (k,2)
            k most probable keypoints.
        selected_descriptors: numpy.ndarray (k,256)
            Descriptors corresponding to the k most probable keypoints.
        """

        def _keep_true_keypoints(points, descriptors, H, shape):
            """ Keep only the points whose warped coordinates by H are still inside shape. """
            warped_points = warp_keypoints(points[:, :2], H)
            mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) & \
                   (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
            return points[mask, :], descriptors[mask, :]

        selected_keypoints, selected_descriptors = _keep_true_keypoints(keypoints, descriptors, H, shape)
        '''
        selected_keypoints, selected_descriptors = select_k_best(selected_keypoints, selected_descriptors,
                                                                 keep_k_points)
        '''
        return selected_keypoints, selected_descriptors

    def h_mtx_estimation_benchmark_upd(self, fnames, resize_480x640=True):
        res = {'i': {'t1': [], 't3': [], 't5': []},
               'v': {'t1': [], 't3': [], 't5': []}}

        # Re-create local detector-descriptor
        self.cfg.task.task_params.detector.max_keypoints_480x640 = 1000
        self.ldd_model = LocalDetectorDescriptor(self.cfg)

        print(f'Let us extract keypoints and local descriptors from images with 640x480')
        self.ldd_model.evaluate(fnames,
                                bbxs=None,
                                resize_480x640=resize_480x640)
        fname_prefix = f'wh_480x640.pkl'
        output_shape_wh = (640, 480)

        seq_names = os.listdir(self.hpsequences_path)
        for _, seq_name in enumerate(tqdm(seq_names, total=len(seq_names))):
            if seq_name in set(self.outliers):
                continue

            with open(osp.join(self.preds, seq_name, f'1_{fname_prefix}'), 'rb') as f:
                data = pickle.load(f)
            keypoints_a, descriptor_a = data['kpts'], data['descs']

            img1 = cv2.imread(os.path.join(self.hpsequences_path, seq_name, "1.ppm"), -1)

            for img_id in range(2, 7):
                with open(osp.join(self.preds, seq_name, f'{img_id}_{fname_prefix}'), 'rb') as f:
                    data = pickle.load(f)
                keypoints_b, descriptor_b = data['kpts'], data['descs']

                img2 = cv2.imread(os.path.join(self.hpsequences_path, seq_name, str(img_id) + ".ppm"), -1)

                h_gt = np.loadtxt(os.path.join(self.hpsequences_path, seq_name, "H_1_" + str(img_id)))

                h_gt = scale_homography(h_gt, img1.shape[:2][::-1], new_scale=output_shape_wh, pre=False)
                h_gt = scale_homography(h_gt, img2.shape[:2][::-1], new_scale=output_shape_wh, pre=True)

                # Keeps only the points shared between the two views
                kpts_a, descs_a = self._keep_shared_points(keypoints_a,
                                                           descriptor_a,
                                                           h_gt,
                                                           output_shape_wh)

                kpts_b, descs_b = self._keep_shared_points(keypoints_b,
                                                           descriptor_b,
                                                           np.linalg.inv(h_gt),
                                                           output_shape_wh)

                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                matches = bf.match(descs_a, descs_b)
                matches_idx = np.array([m.queryIdx for m in matches])
                m_keypoints = kpts_a[matches_idx, :]
                matches_idx = np.array([m.trainIdx for m in matches])
                m_warped_keypoints = kpts_b[matches_idx, :]

                if m_keypoints.shape[0] < 4 or m_warped_keypoints.shape[0] < 4:
                    res[seq_name[0]]['t1'].append(0)
                    res[seq_name[0]]['t3'].append(0)
                    res[seq_name[0]]['t5'].append(0)
                    continue

                # Estimate the homography between the matches using RANSAC
                H, mask = cv2.findHomography(m_keypoints,
                                             m_warped_keypoints,
                                             cv2.RANSAC,
                                             3,
                                             maxIters=5000)

                if H is None:
                    res[seq_name[0]]['t1'].append(0)
                    res[seq_name[0]]['t3'].append(0)
                    res[seq_name[0]]['t5'].append(0)
                    continue

                # Compute correctness
                corners = np.array([[0, 0, 1],
                                    [0, output_shape_wh[1] - 1, 1],
                                    [output_shape_wh[0] - 1, 0, 1],
                                    [output_shape_wh[0] - 1, output_shape_wh[1] - 1, 1]])
                real_warped_corners = np.dot(corners, np.transpose(h_gt))
                real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                warped_corners = np.dot(corners, np.transpose(H))
                warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]

                mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
                c1 = float(mean_dist <= 1)
                c3 = float(mean_dist <= 3)
                c5 = float(mean_dist <= 5)

                res[seq_name[0]]['t1'].append(c1)
                res[seq_name[0]]['t3'].append(c3)
                res[seq_name[0]]['t5'].append(c5)

        return res

    def matching_score_benchmark(self, fnames, resize_480x640=True):
        res = {'i': [], 'v': []}

        # Re-create local detector-descriptor
        self.cfg.task.task_params.detector.max_keypoints_480x640 = 1000
        self.ldd_model = LocalDetectorDescriptor(self.cfg)

        output_shape_wh = (640, 480)
        print(f'Let us extract keypoints and local descriptors from images with {output_shape_wh}')
        self.ldd_model.evaluate(fnames,
                                bbxs=None,
                                resize_480x640=resize_480x640)
        fname_prefix = f'wh_480x640.pkl'

        seq_names = os.listdir(self.hpsequences_path)
        for _, seq_name in enumerate(tqdm(seq_names, total=len(seq_names))):
            if seq_name in set(self.outliers):
                continue

            with open(osp.join(self.preds, seq_name, f'1_{fname_prefix}'), 'rb') as f:
                data = pickle.load(f)
            keypoints_a, descriptor_a = data['kpts'], data['descs']

            img1 = cv2.imread(os.path.join(self.hpsequences_path, seq_name, "1.ppm"), -1)

            for img_id in range(2, 7):
                with open(osp.join(self.preds, seq_name, f'{img_id}_{fname_prefix}'), 'rb') as f:
                    data = pickle.load(f)
                keypoints_b, descriptor_b = data['kpts'], data['descs']

                img2 = cv2.imread(os.path.join(self.hpsequences_path, seq_name, str(img_id) + ".ppm"), -1)

                h_gt = np.loadtxt(os.path.join(self.hpsequences_path, seq_name, "H_1_" + str(img_id)))

                h_gt = scale_homography(h_gt, img1.shape[:2][::-1], new_scale=output_shape_wh, pre=False)
                h_gt = scale_homography(h_gt, img2.shape[:2][::-1], new_scale=output_shape_wh, pre=True)

                # This part needs to be done with crossCheck=False.
                # All the matched pairs need to be evaluated without any selection.
                bf = cv2.BFMatcher(cv2.NORM_L2)
                matches = bf.match(descriptor_a, descriptor_b)
                matches_idx = np.array([m.queryIdx for m in matches])
                m_keypoints = keypoints_a[matches_idx, :]
                matches_idx = np.array([m.trainIdx for m in matches])
                m_warped_keypoints = keypoints_b[matches_idx, :]

                true_warped_keypoints = warp_keypoints(m_warped_keypoints, np.linalg.inv(h_gt))
                vis_warped = np.all((true_warped_keypoints >= 0) & (true_warped_keypoints <= (np.array(output_shape_wh) - 1)),
                                    axis=-1)
                norm1 = np.linalg.norm(true_warped_keypoints - m_keypoints, axis=-1)

                correct1 = (norm1 < 3)
                count1 = np.sum(correct1 * vis_warped)
                score1 = count1 / np.maximum(np.sum(vis_warped), 1.0)

                matches = bf.match(descriptor_b, descriptor_a)
                matches_idx = np.array([m.queryIdx for m in matches])
                m_warped_keypoints = keypoints_b[matches_idx, :]
                matches_idx = np.array([m.trainIdx for m in matches])
                m_keypoints = keypoints_a[matches_idx, :]

                true_keypoints = warp_keypoints(m_keypoints, h_gt)
                vis = np.all((true_keypoints >= 0) & (true_keypoints <= (np.array(output_shape_wh) - 1)), axis=-1)
                norm2 = np.linalg.norm(true_keypoints - m_warped_keypoints, axis=-1)

                correct2 = (norm2 < 3)
                count2 = np.sum(correct2 * vis)
                score2 = count2 / np.maximum(np.sum(vis), 1.0)

                ms = (score1 + score2) / 2

                res[seq_name[0]].append(ms)

        return res

    def pck_benchmark(self, pck_thresholds):
        pck_res = {"i": {thr: 0 for thr in pck_thresholds},
                   "v": {thr: 0 for thr in pck_thresholds},
                   }

        for seq_name in sorted(os.listdir(self.hpsequences_path)):
            if seq_name in self.outliers:
                continue

            for img_id in range(2, 7):
                pos_a = self.kpts_matches[seq_name + "1_" + str(img_id)]["pos_a"]
                pos_b = self.kpts_matches[seq_name + "1_" + str(img_id)]["pos_b"]

                h_gt = np.loadtxt(osp.join(self.hpsequences_path, seq_name, "H_1_" + str(img_id)))

                pos_a_h = np.concatenate([pos_a, np.ones([pos_a.shape[0], 1])], axis=1)
                pos_b_proj_h = np.dot(h_gt, pos_a_h.T).T
                pos_b_proj = pos_b_proj_h[:, :2] / pos_b_proj_h[:, -1, None]

                dist = np.sqrt(np.sum((pos_b - pos_b_proj[:, :2]) ** 2, axis=1))
                if dist.shape[0] == 0:
                    dist = np.array([float("inf")])

                for thr in pck_thresholds:
                    pck_res["i" if seq_name[0] == "i" else "v"][thr] += np.mean(dist <= thr)
        return pck_res

    def get_dataset_stats(self):
        return {"n_illum_scenes": self.n_illum,
                "n_viewpnt_scenes": self.n_viewpnt,
                "illum_feat": np.array(self.stats_feat_matches["i"]["feat"]),
                "illum_matches": np.array(self.stats_feat_matches["i"]["matches"]),
                "viewpnt_feat": np.array(self.stats_feat_matches["v"]["feat"]),
                "viewpnt_matches": np.array(self.stats_feat_matches["v"]["matches"]),
                }

    def evaluate(self):
        # prepare a dataset (create a list of images: fnames -> seqName_imgName.ppm)
        fnames = []
        folder_name = self.cfg.task.task_params.paths.img_path

        for root, dirs, files in os.walk(folder_name):
            if files:
                fnames_per_scene = [osp.join(root, fname) for fname in files if fname.endswith('ppm')]
                fnames.extend(fnames_per_scene)

        # Let us extract image features
        self.ldd_model.evaluate(fnames)

        # compute matches
        self._get_kpts_and_matches()

        h_bench_out = {int: {"acc_v": float,
                             "acc_i": float,
                             "acc_total": float}}
        pck_bench_out = {int: {"v": float,
                               "i": float,
                               "avg": float}}

        n_i = self.n_illum * 5
        n_v = self.n_viewpnt * 5

        feat_i, feat_v = self.get_dataset_stats()["illum_feat"], self.get_dataset_stats()["viewpnt_feat"]
        matches_i, matches_v = self.get_dataset_stats()["illum_matches"], self.get_dataset_stats()[
            "viewpnt_matches"]
        print('# Features [i]: {:f} - [{:d}, {:d}]'.format(np.mean(feat_i), np.min(feat_i), np.max(feat_i)))
        print('# Features [v]: {:f} - [{:d}, {:d}]'.format(np.mean(feat_v), np.min(feat_v), np.max(feat_v)))
        print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
            (np.sum(matches_i) + np.sum(matches_v)) / (n_i + n_v),
            np.sum(matches_i) / n_i,
            np.sum(matches_v) / n_v)
        )

        # PCK metrics
        pck_thresholds = self.cfg.task.task_params.pck_thresholds
        pck_res = self.pck_benchmark(pck_thresholds)

        for pck_thr in self.cfg.task.task_params.pck_thresholds:
            print("MMA@" + str(pck_thr) + " [v]: ", pck_res["v"][pck_thr] / n_v)
            print("MMA@" + str(pck_thr) + " [i]: ", pck_res["i"][pck_thr] / n_i)
            avg = 0.5 * (pck_res["v"][pck_thr] / n_v + pck_res["i"][pck_thr] / n_i)
            print("MMA@" + str(pck_thr) + " [avg]: ", avg)
            print(11 * "*")
            pck_bench_out[pck_thr] = {"v": pck_res["v"][pck_thr] / n_v,
                                      "i": pck_res["i"][pck_thr] / n_i,
                                      "avg": avg}

        print(22 * '-')

        # Homography metrics
        h_mtx_thresholds = self.cfg.task.task_params.h_mtx_thresholds
        for h_thr in h_mtx_thresholds:
            error_h = self.h_mtx_estimation_benchmark(h_thr)
            print("h_threshold: ", h_thr)
            print("Accuracy (viewpoint): ", np.mean(error_h["v"]))
            print("Accuracy (illumination): ", np.mean(error_h["i"]))
            print("Accuracy total: ", np.mean(error_h["i"] + error_h["v"]))
            print(11 * "*")
            h_bench_out[h_thr] = {"acc_v": np.mean(error_h["v"]),
                                  "acc_i": np.mean(error_h["i"]),
                                  "acc_total": np.mean(error_h["i"] + error_h["v"])}

        h_mtx_res = self.h_mtx_estimation_benchmark_upd(fnames)
        for th in ['t1', 't3', 't5']:
            print(f'th: {np.asarray(h_mtx_res["v"][th]).mean()} / '
                  f'{np.asarray(h_mtx_res["i"][th]).mean()} / '
                  f'{np.asarray(h_mtx_res["i"][th] + h_mtx_res["v"][th]).mean()}')

        print(22 * '-')
        matching_res = self.matching_score_benchmark(fnames)
        print(f'Matching score: ')
        print(f'{np.asarray(matching_res["v"]).mean()} / '
              f'{np.asarray(matching_res["i"]).mean()} / '
              f'{np.asarray(matching_res["i"] + matching_res["v"]).mean()}')

        # write results to the file
        with open(self.cfg.task.task_params.output.res_txt_fname, "w") as f:
            f.write(f"PCK benchmark:\n")
            for pck_thr in pck_thresholds:
                f.write(
                    f"MMA@{pck_thr:d} v/i/avg:"
                    f" {pck_bench_out[pck_thr]['v']:05.3f} / {pck_bench_out[pck_thr]['i']:05.3f} / "
                    f"{pck_bench_out[pck_thr]['avg']:05.3f}\n")

            f.write(f"Homography benchmark:\n")
            for h_thr in h_mtx_thresholds:
                f.write(
                    f"th: {h_thr:d} v/i/avg:"
                    f" {h_bench_out[h_thr]['acc_v']:05.3f} / {h_bench_out[h_thr]['acc_i']:05.3f} / "
                    f"{h_bench_out[h_thr]['acc_total']:05.3f}\n")
