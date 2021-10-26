from PIL import Image
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf
from assets.archs_zoo.r2d2_orig import PatchNet


class HNDesc(PatchNet):
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
        self._add_conv(dim, k=2, stride=2, bn=False, relu=True)
        self.out_dim = dim

    @staticmethod
    def img_transform():
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]
        return tvf.Compose([tvf.ToTensor(),
                            tvf.Normalize(mean=rgb_mean, std=rgb_std)])

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

        preprocess = HNDesc.img_transform()
        net_input = preprocess(img)[None].to(device)
        return net_input

    @staticmethod
    def load_network(model_fn):
        checkpoint = torch.load(model_fn)
        model = HNDesc()
        model.load_state_dict(checkpoint['state_dict'])
        return model.eval()


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    return getattr(m, class_name)


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


class CapsDesc(nn.Module):
    def __init__(self,
                 encoder='resnet50',
                 pretrained=True
                 ):
        super(CapsDesc, self).__init__()
        filters = [256, 512, 1024, 2048]
        resnet = class_for_name("torchvision.models", encoder)(pretrained=pretrained)

        self.firstconv = resnet.conv1  # H/2
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # H/4

        # encoder
        self.layer1 = resnet.layer1  # H/4
        self.layer2 = resnet.layer2  # H/8
        self.layer3 = resnet.layer3  # H/16

        # coarse-level conv
        self.conv_coarse = conv(filters[2], 128, 1, 1)

        # decoder
        self.upconv3 = upconv(filters[2], 512, 3, 2)
        self.iconv3 = conv(filters[1] + 512, 512, 3, 1)
        self.upconv2 = upconv(512, 256, 3, 2)
        self.iconv2 = conv(filters[0] + 256, 256, 3, 1)

        # fine-level conv
        self.conv_fine = conv(256, 128, 1, 1)

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        x = self.firstrelu(self.firstbn(self.firstconv(x)))
        x = self.firstmaxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x_coarse = self.conv_coarse(x3)

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_fine = self.conv_fine(x)

        #return [x_coarse, x_fine]
        return x_fine

    @staticmethod
    def img_transform():
        rgb_mean = [0.485, 0.456, 0.406]
        rgb_std = [0.229, 0.224, 0.225]
        return tvf.Compose([tvf.ToTensor(),
                            tvf.Normalize(mean=rgb_mean, std=rgb_std)])

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

        preprocess = CapsDesc.img_transform()
        net_input = preprocess(img)[None].to(device)
        return net_input

    @staticmethod
    def load_network(model_fn):
        checkpoint = torch.load(model_fn)
        model = CapsDesc()
        model.load_state_dict(checkpoint['state_dict'])
        return model.eval()
