import os
from os import path as osp
import pickle

from tqdm import tqdm
import torch
import torch.nn.functional as F
from assets.archs_zoo.superpoint_orig import SuperPoint
from assets.archs_zoo.r2d2_orig import R2D2orig
from assets.archs_zoo.caps_orig import CAPSNet
from assets.archs_zoo.sift_orig import SIFTModel
from assets.descriptors.hndesc import HNDesc, CapsDesc


class LocalDetectorDescriptor(object):
    def __init__(self, cfg):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.detector, self.descriptor, self.det_and_desc = None, None, None

        # get detector and descriptor names
        self.detector_name = self.cfg.task.task_params.detector.name
        self.descriptor_name = self.cfg.descriptor.descriptor_params.name
        self.desc_backbone = self.cfg.descriptor.descriptor_params.backbone

        self.detector = self._detectors_fabric()

        if self.detector_name == self.descriptor_name:
            self.det_and_desc = self.detector
        else:
            self.descriptor = self._descriptors_fabric().to(self.device)

    def _detectors_fabric(self):
        detector = None
        if self.detector_name == 'sift':
            detector = SIFTModel()
        elif self.detector_name == 'superpoint_orig':
            detector_cfg = self.cfg.task.task_params.detector
            detector = SuperPoint(detector_cfg).to(self.device)
        return detector

    def _descriptors_fabric(self):
        model = None
        descriptor_cfg = self.cfg.descriptor.descriptor_params
        desc_name = descriptor_cfg.name
        desc_backbone = descriptor_cfg.backbone
        if desc_name[:10] == "superpoint":
            model = SuperPoint(descriptor_cfg)
        elif desc_name[:4] == "r2d2":
            model = R2D2orig()
            model = model.load_network(descriptor_cfg.snapshot)
        elif desc_name == "caps":
            model = CAPSNet(descriptor_cfg)
            model.load_network()
            print("CAPS is loaded")
        elif desc_name == 'hndesc':
            if desc_backbone == 'r2d2':
                model = HNDesc()
            elif desc_backbone == 'caps':
                model = CapsDesc()
            model = model.load_network(descriptor_cfg.snapshot)
            print(f'HNDesc model, backbone: {desc_backbone} is loaded')
        return model

    def evaluate_img(self, img_fname):
        if self.det_and_desc is not None:
            resize_max = self.det_and_desc.get_resize_max_img()
        else:
            resize_max = self.detector.get_resize_max_img()

        with torch.no_grad():
            if self.det_and_desc is not None:
                detector_data = self.det_and_desc.img_preprocessing(img_fname,
                                                                    self.device,
                                                                    resize_max)

                preds = self.det_and_desc.forward(detector_data['net_input'])

                if self.detector_name == 'superpoint_orig':
                    kpts = preds['keypoints'][0].detach().data.cpu().numpy()
                    descs = preds['descriptors'][0].t().detach().data.cpu().numpy()
                elif self.detector_name == 'sift':
                    kpts = preds['keypoints'].cpu().numpy()
                    descs = preds['descriptors'].cpu().numpy()
                elif self.detector_name == 'd2net':
                    kpts = preds['keypoints'][:, :2].cpu().numpy()
                    descs = preds['descriptors'].cpu().numpy()

                output = {"kpts": kpts,
                          "descs": descs,
                          'original_size': detector_data['original_size'],
                          'new_size': detector_data['new_size']
                          }
            else:
                detector_data = self.detector.img_preprocessing(img_fname,
                                                                self.device,
                                                                resize_max)
                detector_output = self.detector.forward(detector_data['net_input'])

                # Evaluate descriptor
                descriptor_input = self.descriptor.img_preprocessing(img_fname,
                                                                     self.device,
                                                                     resize_max)

                # a very-very bad code
                kpts = detector_output['keypoints'][0].unsqueeze(0)
                if self.detector_name == 'sift':
                    kpts = detector_output['keypoints'].unsqueeze(0).to(self.device)
                elif self.detector_name == 'd2net':
                    kpts = detector_output['keypoints'][:, :2].unsqueeze(0).to(self.device)

                if self.descriptor_name == 'caps':
                    feat_c, feat_f = self.descriptor.extract_features(descriptor_input.unsqueeze(0),
                                                                      kpts)
                    local_descriptors = torch.cat((feat_c, feat_f), -1).squeeze().t()
                    local_descriptors = F.normalize(local_descriptors, dim=0)
                else:
                    dense_descriptors = self.descriptor.forward(descriptor_input)

                    local_descriptors = CAPSNet.extract_local_descs(descriptor_input,
                                                                    dense_descriptors,
                                                                    kpts)
                    local_descriptors = local_descriptors.squeeze(0).t()

                output = {'kpts': kpts.squeeze(0).detach().data.cpu().numpy(),
                          'descs': local_descriptors.squeeze(0).t().detach().data.cpu().numpy(),
                          'original_size': detector_data['original_size'],
                          'new_size': detector_data['new_size']
                          }
        return output

    def evaluate(self, img_fnames, bbxs=None, resize_480x640=False):
        if self.det_and_desc is not None:
            resize_max = self.det_and_desc.get_resize_max_img()
        else:
            resize_max = self.detector.get_resize_max_img()
        imgs_path_prefix = len(self.cfg.task.task_params.paths.img_home_dir)
        with torch.no_grad():
            for i, img_fname in enumerate(tqdm(img_fnames, total=len(img_fnames))):
                # the last element after split('/') is the image filename
                dst_path = osp.join(self.cfg.task.task_params.output.precomputed_feats_dir,
                                    img_fname[imgs_path_prefix + 1:])

                # let us remove the filename from the path name
                dst_path = dst_path[:dst_path.rfind('/')]
                if not osp.exists(dst_path):
                    os.makedirs(dst_path)

                # define a feature filename (ends with 'pkl')
                feat_fname = img_fname.split('/')[-1][:-4]
                feat_fname = feat_fname + '_wh_480x640.pkl' if resize_480x640 else feat_fname + '.pkl'
                # check if file already exists
                if not osp.exists(osp.join(dst_path, feat_fname)):
                    # Evaluate detector
                    bbx = bbxs[i] if bbxs is not None else None

                    if self.det_and_desc is not None:
                        detector_data = self.det_and_desc.img_preprocessing(img_fname,
                                                                            self.device,
                                                                            resize_max,
                                                                            bbx,
                                                                            resize_480x640)

                        preds = self.det_and_desc.forward(detector_data['net_input'])

                        if self.detector_name == 'superpoint':
                            kpts = preds['keypoints'][0].detach().data.cpu().numpy()
                            descs = preds['descriptors'][0].t().detach().data.cpu().numpy()
                        elif self.detector_name == 'sift':
                            kpts = preds['keypoints'].cpu().numpy()
                            descs = preds['descriptors'].cpu().numpy()
                        elif self.detector_name == 'd2net':
                            kpts = preds['keypoints'][:, :2].cpu().numpy()
                            descs = preds['descriptors'].cpu().numpy()

                        output = {"kpts": kpts,
                                  "descs": descs,
                                  'original_size': detector_data['original_size'],
                                  'new_size': detector_data['new_size']
                                  }
                    else:
                        detector_data = self.detector.img_preprocessing(img_fname,
                                                                        self.device,
                                                                        resize_max,
                                                                        bbx,
                                                                        resize_480x640)
                        detector_output = self.detector.forward(detector_data['net_input'])

                        # Evaluate descriptor
                        descriptor_input = self.descriptor.img_preprocessing(img_fname,
                                                                             self.device,
                                                                             resize_max,
                                                                             bbx,
                                                                             resize_480x640)

                        # a very-very bad code
                        kpts = detector_output['keypoints'][0].unsqueeze(0)
                        if self.detector_name == 'sift':
                            kpts = detector_output['keypoints'].unsqueeze(0).to(self.device)
                        elif self.detector_name == 'd2net':
                            kpts = detector_output['keypoints'][:, :2].unsqueeze(0).to(self.device)

                        if self.descriptor_name == 'caps':
                            feat_c, feat_f = self.descriptor.extract_features(descriptor_input.unsqueeze(0),
                                                                              kpts)
                            local_descriptors = torch.cat((feat_c, feat_f), -1).squeeze().t()
                            local_descriptors = F.normalize(local_descriptors, dim=0)
                        else:
                            dense_descriptors = self.descriptor.forward(descriptor_input)

                            local_descriptors = CAPSNet.extract_local_descs(descriptor_input,
                                                                            dense_descriptors,
                                                                            kpts)
                            local_descriptors = local_descriptors.squeeze(0).t()

                        output = {'kpts': kpts.squeeze(0).detach().data.cpu().numpy(),
                                  'descs': local_descriptors.squeeze(0).t().detach().data.cpu().numpy(),
                                  'original_size': detector_data['original_size'],
                                  'new_size': detector_data['new_size']
                                  }

                    with open(osp.join(dst_path, feat_fname), "wb") as f:
                        pickle.dump(output, f)
