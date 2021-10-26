import numpy as np
import cv2
import torch
import torch.nn.functional as F


def to_homogeneous(points):
    return np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=-1)


def from_homogeneous(points):
    return points[:, :-1] / points[:, -1:]


def compute_homography_error(H, H_gt, h, w):
    corners2 = to_homogeneous(np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]))
    corners1_gt = np.dot(corners2, np.transpose(H_gt))
    corners1_gt = corners1_gt[:, :2] / corners1_gt[:, 2:]

    corners2_DM = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst_DM_GT = cv2.perspectiveTransform(corners2_DM, H_gt).squeeze()

    corners1 = np.dot(corners2, np.transpose(H))
    corners1 = corners1[:, :2] / corners1[:, 2:]
    mean_dist = np.mean(np.linalg.norm(corners1 - corners1_gt, axis=1))
    return mean_dist


def desc_similarity(desc1, desc2):
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return None

    descriptors_a = F.normalize(desc1)
    descriptors_b = F.normalize(desc2)

    sim = torch.sqrt(torch.clamp(2 - 2 * (descriptors_a @ descriptors_b.t()), min=0))
    return sim
