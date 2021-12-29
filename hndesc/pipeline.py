import os
from os import path as osp
from tensorboardX import SummaryWriter
import time
import torch
from tqdm import tqdm
from assets.descriptors.descriptor import LocalDescriptor
from assets.detectors.superpoint_detector import SuperPoint
from assets.archs_zoo.caps_orig import CAPSNet
from data.dataset import MegaDepthDataset
from hndesc.utils import cycle, seed_everything
from data.augmentations import get_img_augmentations
from hndesc.criterion import APCriterion


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        cfg_model = self.cfg.model_params

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        seed_everything(self.cfg.seed)
        self.model = LocalDescriptor(cfg_model)

        n_params = sum(
            p.numel()
            for p in self.model.descriptor.parameters()
            if p.requires_grad
        )
        print(f"Number of network parameters: {n_params}")

        # initialize dataloaders
        self.train_loader, self.val_loader = self._init_dataloaders()
        self.train_loader_iterator = iter(cycle(self.train_loader))

        # create an optimizer
        self.optimizer = torch.optim.Adam(
            self.model.get_model().parameters(), lr=self.cfg.train_params.lr
        )

        # create a scheduler
        cfg_scheduler = self.cfg.train_params.scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=cfg_scheduler.lrate_decay_steps,
            gamma=cfg_scheduler.lrate_decay_factor,
        )

        # create criterion
        self.criterion = APCriterion().to(self.device)

        # create writer (logger)
        self.writer = SummaryWriter(self.cfg.output_params.logger_dir)

        self.start_step = 0
        self.val_total_loss = 1e6
        if self.cfg.model_params.resume_snapshot:
            self._load_model(self.cfg.model_params.resume_snapshot)

    def _init_dataloaders(self):
        cfg_data = self.cfg.data_params
        cfg_train = self.cfg.train_params

        # get image augmentations
        train_img_augs, val_img_augs = get_img_augmentations()

        train_dataset = MegaDepthDataset(
            img_path=osp.join(cfg_data.img_dir, "train"),
            kpts_path=cfg_data.kpts_descs_dir,
            crop_size=cfg_data.crop_size,
            win_size=cfg_data.win_size,
            st_path=osp.join(cfg_data.st_dir, "train"),
            transforms=train_img_augs,
        )

        val_dataset = MegaDepthDataset(
            img_path=osp.join(cfg_data.img_dir, "test"),
            kpts_path=cfg_data.kpts_descs_dir,
            crop_size=cfg_data.crop_size,
            win_size=cfg_data.win_size,
            transforms=val_img_augs,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg_train.bs,
            shuffle=True,
            pin_memory=True,
            num_workers=cfg_train.n_workers,
            drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg_train.bs,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg_train.n_workers,
            drop_last=True,
        )
        return train_loader, val_loader

    def _get_local_descs(self, data_sample):
        # dense features for a given image crop
        descs_anc_crop = self.model.forward(
            data_sample["crop_src"].to(self.device)
        )
        descs_pos_crop = self.model.forward(
            data_sample["crop_trg"].to(self.device)
        )
        """
        descs_hn_crop = self.model.forward(
            data_sample["crop_hn"].to(self.device)
        )
        mask_valid_kpts_hn = (
            data_sample["mask_valid_kpts_hn"].squeeze(-1).to(self.device)
        )
        kpts_hn = data_sample["crop_kpts_hn"]
        """

        mask_valid_kpts = (
            data_sample["mask_valid_kpts"].squeeze(-1).to(self.device)
        )

        kpts_crop_ids = torch.sum(mask_valid_kpts, dim=1)

        kpts_src = data_sample["crop_src_kpts"]
        kpts_trg = data_sample["crop_trg_kpts"]

        if self.cfg.model_params.backbone_net == "caps":
            loc_desc_anc = CAPSNet.extract_local_descs(
                data_sample["crop_src"],
                descs_anc_crop,
                kpts_src.to(self.device),
            )
            loc_desc_pos = CAPSNet.extract_local_descs(
                data_sample["crop_src"],
                descs_pos_crop,
                kpts_trg.to(self.device),
            )
            """
            loc_desc_hn = CAPSNet.extract_local_descs(
                data_sample["crop_src"], descs_hn_crop, kpts_hn.to(self.device)
            )
            """
        else:
            # local descriptors (anchors)
            loc_desc_anc = SuperPoint.sample_descriptors_window_batch(
                kpts=data_sample["kpts_src_win"].to(self.device),
                descs=descs_anc_crop.to(self.device),
                window=self.cfg.data_params.win_size,
            )
            # local descriptors (positives)
            loc_desc_pos = SuperPoint.sample_descriptors_window_batch(
                kpts=data_sample["kpts_trg_win"].to(self.device),
                descs=descs_pos_crop.to(self.device),
                window=self.cfg.data_params.win_size,
            )
            """
            # local descriptors (negatives)
            loc_desc_hn = SuperPoint.sample_descriptors_int_batch(kpts=kpts_hn.to(self.device),
                                                                  descs=descs_hn_crop.to(self.device),
                                                                  reverse=False)
            """

        # let us apply validity mask
        loc_desc_anc = loc_desc_anc[mask_valid_kpts]
        loc_desc_pos = loc_desc_pos[mask_valid_kpts]
        # loc_desc_hn = loc_desc_hn[mask_valid_kpts_hn]

        return loc_desc_anc, loc_desc_pos, kpts_crop_ids  # , loc_desc_hn

    def _save_model(self, step, loss_val, best_val=False):
        if not osp.exists(self.cfg.output_params.snapshot_dir):
            os.makedirs(self.cfg.output_params.snapshot_dir)

        fname_out = (
            "best_val.pth" if best_val else "snapshot{:06d}.pth".format(step)
        )
        save_path = osp.join(self.cfg.output_params.snapshot_dir, fname_out)
        model_state = self.model.get_model().state_dict()
        torch.save(
            {
                "step": step,
                "state_dict": model_state,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "val_loss": loss_val,
            },
            save_path,
        )

    def _load_model(self, snapshot):
        data_dict = torch.load(snapshot)
        self.model.get_model().load_state_dict(data_dict["state_dict"])
        self.optimizer.load_state_dict(data_dict["optimizer"])
        self.scheduler.load_state_dict(data_dict["scheduler"])
        self.start_step = data_dict["step"]
        if "val_loss" in data_dict:
            self.val_total_loss = data_dict["val_loss"]

    def _train_batch(self):
        train_sample = next(self.train_loader_iterator)
        (loc_desc_anc, loc_desc_pos, kpts_crop_ids) = self._get_local_descs(
            train_sample
        )

        self.optimizer.zero_grad()

        # compute loss
        loss, ap = self.criterion(loc_desc_anc, loc_desc_pos, kpts_crop_ids)
        loss.backward()

        # update the optimizer
        self.optimizer.step()

        # update the scheduler
        self.scheduler.step()
        return loss.item(), ap.item()

    def _validate(self):
        self.model.get_model().eval()
        loss_value = 0.0
        ap_value = 0.0
        with torch.no_grad():
            for val_sample in tqdm(self.val_loader):
                (
                    loc_desc_anc,
                    loc_desc_pos,
                    kpts_crop_ids,
                ) = self._get_local_descs(val_sample)

                # compute loss
                loss, ap = self.criterion(
                    loc_desc_anc, loc_desc_pos, kpts_crop_ids
                )
                """
                loss, ap = self.criterion(
                    loc_desc_anc, loc_desc_pos, loc_desc_hn, kpts_crop_ids
                )
                """

                loss_value += loss.item()
                ap_value += ap.item()

        self.model.get_model().train()
        return loss_value / len(self.val_loader), ap_value / len(
            self.val_loader
        )

    def run(self):
        print(f"Start training. Step: {self.start_step}")
        train_start_time = time.time()
        train_log_iter_time = time.time()
        for step in range(
            self.start_step + 1,
            self.start_step + self.cfg.train_params.n_train_iters,
        ):
            train_loss_batch, ap_batch = self._train_batch()

            if (
                step % self.cfg.output_params.log_scalar_interval == 0
                and step > 0
            ):
                self.writer.add_scalar(
                    "Train_total_loss_batch", train_loss_batch, step
                )
                print(
                    f"Elapsed time [min] for {self.cfg.output_params.log_scalar_interval} training iterations: "
                    f"{(time.time() - train_log_iter_time) / 60.}"
                )
                train_log_iter_time = time.time()

                print(
                    f"Step {step} out of {self.cfg.train_params.n_train_iters} is done. Train loss (per batch): "
                    f"{train_loss_batch}. AP: {ap_batch}"
                )

            if (
                step % self.cfg.output_params.validate_interval == 0
                and step > 0
            ):
                print(f"Perform validation. Step: {step}")
                val_time = time.time()
                best_val = False
                val_loss, ap_val = self._validate()
                self.writer.add_scalar("Val_total_loss", val_loss, step)
                self.writer.add_scalar("Val_total_AP", ap_val, step)
                if val_loss < self.val_total_loss:
                    self.val_total_loss = val_loss
                    best_val = True
                self._save_model(step, val_loss, best_val=best_val)
                print(f"Validation loss: {val_loss}, ap: {ap_val}")
                print(
                    f"Elapsed time [min] for validation: {(time.time() - val_time) / 60.}"
                )
                train_log_iter_time = time.time()

        print(
            f"Elapsed time for training [min] {(time.time() - train_start_time) / 60.}"
        )
        print("Done")
