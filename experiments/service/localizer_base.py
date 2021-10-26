import numpy as np
import os
from os import path as osp
import torch
import shutil
from experiments.service.matchers_factory import MatchersFactory


class Camera:
    def __init__(self):
        self.camera_model = None
        self.intrinsics = None
        self.qvec = None
        self.t = None

    def set_intrinsics(self, camera_model, intrinsics):
        self.camera_model = camera_model
        self.intrinsics = intrinsics

    def set_pose(self, qvec, t):
        self.qvec = qvec
        self.t = t


def quaternion_to_rotation_matrix(qvec):
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    R = np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                  [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
                  [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y]])
    return R


def camera_center_to_translation(c, qvec):
    R = quaternion_to_rotation_matrix(qvec)
    return (-1) * np.matmul(R, c)


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2


class Localizer(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matcher = MatchersFactory(self.cfg.matcher).get_matcher()

        if self.cfg.colmap_data.db_fname:
            self.target_database = osp.join(self.cfg.paths.loc_res_dir,
                                            self.cfg.colmap_data.db_fname.split('/')[-1])

            print("Copying the target database...")
            if osp.exists(self.target_database):
                os.remove(self.target_database)
            shutil.copyfile(self.cfg.colmap_data.db_fname, self.target_database)

        print("Copying the target database... Done!")

        self.camera_parameters = {}
        self.images = {}
        self.cameras = {}

    def localize(self):
        raise NotImplementedError
