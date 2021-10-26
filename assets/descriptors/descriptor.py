import torch
from assets.descriptors.hndesc import HNDesc, CapsDesc


class LocalDescriptor(object):
    def __init__(self, cfg):
        self.cfg = cfg
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.descriptor = self._descriptor_factory().to(device)

    def _descriptor_factory(self):
        descriptor = None
        if self.cfg.backbone_net == 'r2d2':
            descriptor = HNDesc()
        elif self.cfg.backbone_net == 'caps':
            descriptor = CapsDesc()
        return descriptor

    def get_img_transform(self):
        return self.descriptor.img_transform()

    def forward(self, x):
        return self.descriptor.forward(x)

    def get_model(self):
        return self.descriptor
