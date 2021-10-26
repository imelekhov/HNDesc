import os
from os import path as osp
import shutil
import torch
from experiments.service.matchers_factory import MatchersFactory
from experiments.service.ldd_factory import LocalDetectorDescriptor


class BaseRetrievalAgent(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.retrieval_agent_cfg = self.cfg.task.task_params

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Warning! If the output directory exists, we will delete it
        if osp.exists(self.retrieval_agent_cfg.output.det_desc_home_dir):
            shutil.rmtree(self.retrieval_agent_cfg.output.det_desc_home_dir)

        # Create an empty output directory
        os.makedirs(self.retrieval_agent_cfg.output.det_desc_home_dir)

        # Matcher
        self.matcher = MatchersFactory(self.cfg.matcher).get_matcher()

        # Local detector-descriptor
        self.ldd_model = LocalDetectorDescriptor(self.cfg)

    def retrieve(self):
        raise NotImplementedError
