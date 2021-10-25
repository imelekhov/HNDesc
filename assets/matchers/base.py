import torch


class BaseMatcher(object):
    def __init__(self, matcher_cfg):
        self.matcher_cfg = matcher_cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_fails = 0

    def match(self, s1: torch.Tensor, s2: torch.Tensor):
        raise NotImplementedError
