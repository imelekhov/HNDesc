from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf


class PatchNet(nn.Module):
    """ Helper class to construct a fully-convolutional network that
        extract a l2-normalized patch descriptor.
    """

    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        super().__init__()
        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True):
        d = self.dilation * dilation
        if self.dilated:
            conv_params = dict(padding=((k - 1) * d) // 2, dilation=d, stride=1)
            self.dilation *= stride
        else:
            conv_params = dict(padding=((k - 1) * d) // 2, dilation=d, stride=stride)

        self.ops.append(nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params))
        if bn and self.bn: self.ops.append(self._make_bn(outd))
        if relu: self.ops.append(nn.ReLU(inplace=True))
        self.curchan = outd

    def forward(self, x):
        assert self.ops, "You need to add convolutions first"
        for n, op in enumerate(self.ops):
            x = op(x)
        return F.normalize(x, p=2, dim=1)


class L2_Net(PatchNet):
    """ Compute a 128D descriptor for all overlapping 32x32 patches.
        From the L2Net paper (CVPR'17).
    """
    def __init__(self, dim=128, **kw):
        PatchNet.__init__(self, **kw)
        add_conv = lambda n, **kw: self._add_conv(n, **kw)
        add_conv(32)
        add_conv(32)
        add_conv(64, stride=2)
        add_conv(64)
        add_conv(128, stride=2)
        add_conv(128)
        add_conv(128, k=7, stride=8, bn=False, relu=False)
        self.out_dim = dim


class Quad_L2Net (PatchNet):
    """ Same than L2_Net, but replace the final 8x8 conv by 3 successive 2x2 convs.
    """
    def __init__(self, dim=128, mchan=4, relu22=False, **kw ):
        PatchNet.__init__(self, **kw)
        self._add_conv(  8*mchan)
        self._add_conv(  8*mchan)
        self._add_conv( 16*mchan, stride=2)
        self._add_conv( 16*mchan)
        self._add_conv( 32*mchan, stride=2)
        self._add_conv( 32*mchan)
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv( 32*mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        self.out_dim = dim


class Quad_L2Net_ConfCFS (Quad_L2Net):
    """ Same than Quad_L2Net, with 2 confidence maps for repeatability and reliability.
    """
    def __init__(self, **kw ):
        Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1) 

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        # compute the confidence maps
        ureliability = self.clf(x**2)
        urepeatability = self.sal(x**2)
        return self.normalize(x, ureliability, urepeatability)

    @staticmethod
    def img_preprocessing(fname, device, resize_max=None, bbxs=None, resize_480x640=False):
        img = Image.open(fname).convert('RGB')

        if resize_480x640:
            img = img.resize((640, 480))

        if bbxs is not None:
            img = img.crop(bbxs)

        w, h = img.size

        if resize_max and max(w, h) > resize_max:
            scale = resize_max / max(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            img = img.resize((w_new, h_new))

        preprocess = tvf.Compose([tvf.ToTensor(),
                                  tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        net_input = preprocess(img)[None].to(device)
        return net_input


class R2D2orig(PatchNet):
    """ Same than L2_Net, but replace the final 8x8 conv by 3 successive 2x2 convs.
    """
    def __init__(self, dim=128, mchan=4, relu22=False, **kw):
        PatchNet.__init__(self, **kw)
        self._add_conv(8 * mchan)
        self._add_conv(8 * mchan)
        self._add_conv(16 * mchan, stride=2)
        self._add_conv(16 * mchan)
        self._add_conv(32 * mchan, stride=2)
        self._add_conv(32 * mchan)

        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv(32 * mchan, k=2, stride=2, relu=relu22)
        self._add_conv(32 * mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        self.out_dim = dim

    @staticmethod
    def load_network(model_fn):
        checkpoint = torch.load(model_fn)
        print(f">> Creating network: {checkpoint['net']}")
        net = eval(checkpoint['net'])

        # initialization
        weights = checkpoint['state_dict']
        net.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
        print(f">> The network: {checkpoint['net']} has been loaded")
        return net.eval()

    '''
    @staticmethod
    def img_preprocessing(fname, device):
        img = Image.open(fname).convert('RGB')
        preprocess = tvf.Compose([tvf.ToTensor(),
                                  tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        net_input = preprocess(img).to(device)
        return net_input
    '''

    @staticmethod
    def img_preprocessing(fname, device, resize_max=None, bbxs=None, resize_480x640=False):
        img = Image.open(fname).convert('RGB')

        if resize_480x640:
            img = img.resize((640, 480))

        if bbxs is not None:
            img = img.crop(bbxs)

        w, h = img.size

        if resize_max and max(w, h) > resize_max:
            scale = resize_max / max(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            img = img.resize((w_new, h_new))

        preprocess = tvf.Compose([tvf.ToTensor(),
                                  tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        net_input = preprocess(img).to(device)
        return net_input
