import numpy as np
import collections
import os
from os import path as osp
import sqlite3
import pickle
from tqdm import tqdm
import time
import subprocess
import torch
import shutil
from experiments.localization.localizers.base import (
    ColmapLoclalizerBase,
    Camera,
    camera_center_to_translation,
    image_ids_to_pair_id,
)
from assets.archs_zoo.superpoint_orig import SuperPoint


class AachenLocalizer(ColmapLoclalizerBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.paths = self.cfg.task.task_params.paths
        self.colmap_data = self.cfg.task.task_params.colmap_data
        self.output = self.cfg.task.task_params.output
        self.precomputed_feats_dir = self.output.precomputed_feats_dir

    def _create_list_of_imgs(self):
        start_time = time.time()
        imgs_dir = self.cfg.task.task_params.paths.img_path
        print(f"Creating a list of images for feature extraction... ")
        img_fnames = []
        pairs_fname = self.colmap_data.image_pairs_fname
        with open(pairs_fname, "r") as f:
            for line in f:
                item1, item2 = line.rstrip().split(" ")
                img_fnames.append(osp.join(imgs_dir, item1))
                img_fnames.append(osp.join(imgs_dir, item2))
        print(
            f"Creating a list of images for feature extraction... Done! Elapsed time, s: {time.time() - start_time}"
        )
        return list(set(img_fnames))

    def _preprocess_reference_model(self):
        print("Preprocessing the reference detector...")

        # Recover intrinsics.
        with open(self.colmap_data.intrinsics) as f:
            raw_intrinsics = f.readlines()

        for intrinsics in raw_intrinsics:
            intrinsics = intrinsics.strip("\n").split(" ")

            image_name = intrinsics[0]
            camera_model = intrinsics[1]
            intrinsics = [float(param) for param in intrinsics[2:]]

            camera = Camera()
            camera.set_intrinsics(
                camera_model=camera_model, intrinsics=intrinsics
            )

            self.camera_parameters[image_name] = camera

        # Recover poses.
        with open(self.colmap_data.nvm_database) as f:
            raw_extrinsics = f.readlines()

        # Skip the header.
        n_cameras = int(raw_extrinsics[2])
        raw_extrinsics = raw_extrinsics[3 : 3 + n_cameras]

        for extrinsics in raw_extrinsics:
            extrinsics = extrinsics.strip("\n").split(" ")

            image_name = extrinsics[0]

            # Skip the focal length. Skip the distortion and terminal 0.
            qw, qx, qy, qz, cx, cy, cz = [
                float(param) for param in extrinsics[2:-2]
            ]

            qvec = np.array([qw, qx, qy, qz])
            c = np.array([cx, cy, cz])

            # NVM -> COLMAP.
            t = camera_center_to_translation(c, qvec)

            self.camera_parameters[image_name].set_pose(qvec=qvec, t=t)

        return self.camera_parameters

    def _recover_database_images_and_ids(self):

        # Connect to the database.
        connection = sqlite3.connect(self.target_database)
        cursor = connection.cursor()

        # Recover database images and ids.
        cursor.execute("SELECT name, image_id, camera_id FROM images;")
        for row in cursor:
            self.images[row[0]] = row[1]
            self.cameras[row[0]] = row[2]

        # Close the connection to the database.
        cursor.close()
        connection.close()

        return self.images, self.cameras

    def _generate_empty_reconstruction(self):
        print("Generating the empty reconstruction...")

        with open(osp.join(self.output.loc_res_dir, "cameras.txt"), "w") as f:
            for image_name in self.images:
                image_id = self.images[image_name]
                camera_id = self.cameras[image_name]
                try:
                    camera = self.camera_parameters[image_name]
                except:
                    continue
                f.write(
                    "%d %s %s\n"
                    % (
                        camera_id,
                        camera.camera_model,
                        " ".join(map(str, camera.intrinsics)),
                    )
                )

        with open(osp.join(self.output.loc_res_dir, "images.txt"), "w") as f:
            for image_name in self.images:
                image_id = self.images[image_name]
                camera_id = self.cameras[image_name]
                try:
                    camera = self.camera_parameters[image_name]
                except:
                    continue
                f.write(
                    "%d %s %s %d %s\n\n"
                    % (
                        image_id,
                        " ".join(map(str, camera.qvec)),
                        " ".join(map(str, camera.t)),
                        camera_id,
                        image_name,
                    )
                )

        with open(osp.join(self.output.loc_res_dir, "points3D.txt"), "w") as f:
            pass

    def _import_features(self):
        # Connect to the database.
        connection = sqlite3.connect(self.target_database)
        cursor = connection.cursor()

        cursor.execute("DELETE FROM keypoints;")
        cursor.execute("DELETE FROM descriptors;")
        cursor.execute("DELETE FROM matches;")
        connection.commit()

        # Import the features.
        print("Importing features...")

        for image_name, image_id in tqdm(
            self.images.items(), total=len(self.images.items())
        ):
            feat_fname = image_name[:-3] + "pkl"
            det_data_fname = osp.join(self.precomputed_feats_dir, feat_fname)

            if not osp.isfile(det_data_fname):
                continue

            with open(det_data_fname, "rb") as f:
                data = pickle.load(f)

            # let us rescale keypoint coordinates
            keypoints = SuperPoint.rescale_kpts(
                data["kpts"],
                rescaled_size=data["new_size"],
                original_size=data["original_size"],
            )

            # keypoints = data["kpts"]
            n_keypoints, kpt_dim = keypoints.shape[0], keypoints.shape[1]

            if kpt_dim < self.max_kpt_dim:
                keypoints = np.concatenate(
                    [
                        keypoints,
                        np.zeros((n_keypoints, self.max_kpt_dim - kpt_dim)),
                    ],
                    axis=1,
                ).astype(np.float32)

            keypoints_str = keypoints.tostring()
            cursor.execute(
                "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                (
                    image_id,
                    keypoints.shape[0],
                    keypoints.shape[1],
                    keypoints_str,
                ),
            )
            connection.commit()

        # Close the connection to the database.
        cursor.close()
        connection.close()

    def _match_features(self):
        # Connect to the database.
        connection = sqlite3.connect(self.target_database)
        cursor = connection.cursor()

        # Match the features and insert the matches in the database.
        print("Matching...")
        # print("Features distance threshold: ", args.feat_dist_threshold)

        with open(self.colmap_data.image_pairs_fname, "r") as f:
            raw_pairs = f.readlines()

        image_pair_ids = set()

        for raw_pair in tqdm(raw_pairs, total=len(raw_pairs)):
            image_name1, image_name2 = raw_pair.strip("\n").split(" ")
            if (image_name1 not in self.images) or (
                image_name2 not in self.images
            ):
                continue

            with open(
                osp.join(self.precomputed_feats_dir, image_name1[:-3] + "pkl"),
                "rb",
            ) as f:
                data1 = pickle.load(f)

            with open(
                osp.join(self.precomputed_feats_dir, image_name2[:-3] + "pkl"),
                "rb",
            ) as f:
                data2 = pickle.load(f)

            desc1, desc2 = data1["descs"], data2["descs"]

            matches = self.matcher.match(
                torch.from_numpy(desc1).to(self.device),
                torch.from_numpy(desc2).to(self.device),
            )

            image_id1, image_id2 = (
                self.images[image_name1],
                self.images[image_name2],
            )
            image_pair_id = image_ids_to_pair_id(image_id1, image_id2)

            if image_pair_id in image_pair_ids:
                continue
            image_pair_ids.add(image_pair_id)

            if image_id1 > image_id2:
                matches = matches[:, [1, 0]]

            matches_str = matches.tostring()
            cursor.execute(
                "INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                (
                    image_pair_id,
                    matches.shape[0],
                    matches.shape[1],
                    matches_str,
                ),
            )
            connection.commit()

        # Close the connection to the database.
        cursor.close()
        connection.close()
        print(
            "Number of pairs where the number of matches was too small: ",
            self.matcher.n_fails,
        )

    def _geometric_verification(self):
        print("Running geometric verification...")

        subprocess.call(
            [
                osp.join(self.paths.colmap_dir, "colmap"),
                "matches_importer",
                "--database_path",
                self.target_database,
                "--match_list_path",
                self.colmap_data.image_pairs_fname,
                "--match_type",
                "pairs",
            ]
        )

    def _reconstruct(self):
        database_model_path = osp.join(
            self.output.loc_res_dir, "sparse-database"
        )

        if osp.exists(database_model_path):
            shutil.rmtree(database_model_path)
        os.makedirs(database_model_path)

        # CONFIG_0 (by D2-Net for Extended-CMU)
        subprocess.call(
            [
                osp.join(self.paths.colmap_dir, "colmap"),
                "point_triangulator",
                "--database_path",
                self.target_database,
                "--image_path",
                self.paths.img_path,
                "--input_path",
                self.output.loc_res_dir,
                "--output_path",
                database_model_path,
                "--Mapper.min_num_matches",
                "4",
                "--Mapper.init_min_num_inliers",
                "4",
                "--Mapper.abs_pose_min_num_inliers",
                "4",
                "--Mapper.abs_pose_min_inlier_ratio",
                "0.05",
                "--Mapper.ba_local_max_num_iterations",
                "50",
                "--Mapper.abs_pose_max_error",
                "20",
                "--Mapper.filter_max_reproj_error",
                "12",
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_principal_point",
                "0",
                "--Mapper.ba_refine_extra_params",
                "0",
            ]
        )

    def _register_queries(self):
        final_model_path = osp.join(self.output.loc_res_dir, "sparse-final")

        if osp.exists(final_model_path):
            shutil.rmtree(final_model_path)
        os.makedirs(final_model_path)

        # Register the query images.
        # CONFIG_0 (by D2-Net for Extended-CMU)
        subprocess.call(
            [
                osp.join(self.paths.colmap_dir, "colmap"),
                "image_registrator",
                "--database_path",
                self.target_database,
                "--input_path",
                osp.join(self.output.loc_res_dir, "sparse-database"),
                "--output_path",
                final_model_path,
                "--Mapper.min_num_matches",
                "4",
                "--Mapper.init_min_num_inliers",
                "4",
                "--Mapper.abs_pose_min_num_inliers",
                "4",
                "--Mapper.abs_pose_min_inlier_ratio",
                "0.05",
                "--Mapper.ba_local_max_num_iterations",
                "50",
                "--Mapper.abs_pose_max_error",
                "20",
                "--Mapper.filter_max_reproj_error",
                "12",
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_principal_point",
                "0",
                "--Mapper.ba_refine_extra_params",
                "0",
            ]
        )

    def _recover_query_poses(self):
        print("Recovering query poses...")

        final_txt_model_path = osp.join(
            self.output.loc_res_dir, "sparse-final-txt"
        )

        if osp.exists(final_txt_model_path):
            shutil.rmtree(final_txt_model_path)
        os.makedirs(final_txt_model_path)

        # Convert the detector to TXT.
        subprocess.call(
            [
                osp.join(self.paths.colmap_dir, "colmap"),
                "model_converter",
                "--input_path",
                osp.join(self.output.loc_res_dir, "sparse-final"),
                "--output_path",
                final_txt_model_path,
                "--output_type",
                "TXT",
            ]
        )

        # Recover query names.
        query_image_list_path = self.colmap_data.queries_to_localize
        with open(query_image_list_path) as f:
            raw_queries = f.readlines()

        query_names = set()
        for raw_query in raw_queries:
            raw_query = raw_query.strip("\n").split(" ")
            query_name = raw_query[0]
            query_names.add(query_name)

        with open(osp.join(final_txt_model_path, "images.txt")) as f:
            raw_extrinsics = f.readlines()

        f = open(self.output.res_txt_fname, "w")

        # Skip the header.
        for extrinsics in raw_extrinsics[4::2]:
            extrinsics = extrinsics.strip("\n").split(" ")

            image_name = extrinsics[-1]

            if image_name in query_names:
                # Skip the IMAGE_ID ([0]), CAMERA_ID ([-2]), and IMAGE_NAME ([-1]).
                f.write(
                    "%s %s\n"
                    % (image_name.split("/")[-1], " ".join(extrinsics[1:-2]))
                )

        f.close()

    def localize(self):

        img_fnames = self._create_list_of_imgs()

        # Let us extract image features
        self.ldd_model.evaluate(img_fnames)

        # Reconstruction pipeline
        _ = self._preprocess_reference_model()
        print("Recover images from colmap db...")
        _, _ = self._recover_database_images_and_ids()

        self._generate_empty_reconstruction()

        # Import features
        self._import_features()

        # Match features
        self._match_features()
        # Geometric verification
        self._geometric_verification()

        self._reconstruct()
        self._register_queries()

        self._recover_query_poses()

        print("Done")
