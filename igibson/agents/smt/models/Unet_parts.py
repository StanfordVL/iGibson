import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels


class Scaler(nn.Module):
    def __init__(self, nch, outsize, bilinear=False, norm='batchnorm'):
        """TODO: to be defined1. """
        nn.Module.__init__(self)
        self.nch = nch
        self.outsize = outsize
        max_out_scale = np.amax(self.outsize)
        n_upscale = int(np.ceil(math.log(max_out_scale, 2)))
        self.scaler = nn.ModuleList([
            UNetUp(nch, nch, bilinear=bilinear, norm=norm)
            for i in range(n_upscale)
        ])

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        for mod in self.scaler:
            x = mod(x)
        return x


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, norm='batchnorm'):
        super(double_conv, self).__init__()
        if norm == 'batchnorm':
            normlayer = nn.BatchNorm2d
        elif norm == 'instancenorm':
            normlayer = nn.InstanceNorm2d
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            normlayer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            normlayer(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batchnorm'):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, norm=norm)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, norm='batchnorm'):
        super(down, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mpconv = nn.Sequential(nn.MaxPool2d(2),
                                    double_conv(in_ch, out_ch, norm=norm))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UNetUp(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 bilinear=True,
                 skip=False,
                 norm='batchnorm',
                 scale_factor=2):
        super(UNetUp, self).__init__()

        if bilinear:
            self.upscale = nn.Upsample(scale_factor=scale_factor,
                                       mode='bilinear',
                                       align_corners=True)
        else:
            self.upscale = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        if skip:
            self.conv = double_conv(in_ch + out_ch, out_ch, norm=norm)
        else:
            self.conv = double_conv(in_ch, out_ch, norm=norm)

    def forward(self, x1, x2=None):
        x = self.upscale(x1)
        if x2 is not None:
            diffY = x2.size()[2] - x.size()[2]
            diffX = x2.size()[3] - x.size()[3]
            x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2,
                          diffY - diffY // 2))
            x = torch.cat([x, x2], 1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


# =============== Component modules ====================


class UNetEncoder(nn.Module):
    def __init__(self,
                 n_channels,
                 nsf=16,
                 n_downscale=4,
                 norm='batchnorm',
                 **kwargs):
        super().__init__()
        self.inc = inconv(n_channels, nsf, norm=norm)
        self.down_mods = nn.ModuleList([
            down(nsf * (2**min(i, 3)), nsf * (2**min(i + 1, 3)), norm=norm)
            for i in range(n_downscale)
        ])

    def forward(self, x, *args, **kwargs):
        x1 = self.inc(x)  # (bs, nsf, ..., ...)
        out_x = [x1]
        for mod in self.down_mods:
            out_x.append(mod(out_x[-1]))

        return out_x  # {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}


class UNetDecoder(nn.Module):
    def __init__(self,
                 n_classes=2,
                 nsf=16,
                 n_downscale=4,
                 n_upscale=4,
                 in_scale=np.array([32, 32]),
                 out_scale=np.array([64, 64]),
                 norm='batchnorm',
                 **kwargs):
        super().__init__()
        self.n_upscale = n_upscale
        self.in_scale = in_scale
        self.out_scale = out_scale
        up_mods = [
            UNetUp(nsf * (2**np.clip(i + 1, 0, 3)),
                   nsf * (2**np.clip(i, 0, 3)),
                   skip=True,
                   norm=norm) for i in range(n_downscale - 1, -1, -1)
        ]
        up_mods = up_mods + [
            UNetUp(nsf, nsf, skip=False, norm=norm)
            for i in range(n_upscale - n_downscale)
        ]
        self.up_mods = nn.ModuleList(up_mods)

        self.outc = outconv(nsf, n_classes)
        self.nsf = nsf
        self.n_classes = n_classes

    def forward(self, xin, *args, **kwargs):
        """
        xin is a dictionary that consists of x1, x2, x3, x4, x5 keys
        from the UNetEncoder
        """

        x = xin.pop()
        for mod in self.up_mods:
            if len(xin):
                x = mod(x, xin.pop())
            else:
                x = mod(x)

        x = self.outc(x)  # (bs, n_classes, ..., ...)
        x = x[:, :, (x.shape[-2] // 2) -
              (self.out_scale[0] // 2):(x.shape[-2] // 2) +
              (self.out_scale[0] // 2) +
              (self.out_scale[0] % 2), (x.shape[-1] // 2) -
              (self.out_scale[1] // 2):(x.shape[-1] // 2) +
              (self.out_scale[1] // 2) + (self.out_scale[1] % 2)]

        return x


class ResNetRGBEncoderByProjection(nn.Module):
    """
    Encodes RGB image via ResNet block1, block2 and merges them.
    """
    def __init__(self,
                 pretrained,
                 resnet_type='resnet50',
                 out_scale=(32, 32),
                 norm='batchnorm'):
        super().__init__()
        print('=====> ResNetRGBEncoder - pretrained: {}'.format(pretrained))
        if resnet_type == 'resnet50':
            resnet = tmodels.resnet50(pretrained=pretrained)
        elif resnet_type == 'resnet18':
            resnet = tmodels.resnet18(pretrained=pretrained)
        else:
            raise ValueError(f'ResNet type {resnet_type} not defined!')

        self.resnet_base = nn.Sequential(  # (B, 3, H, W)
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.resnet_block1 = resnet.layer1  # (256, H/4, W/4)
        self.resnet_block2 = resnet.layer2  # (512, H/8, W/8)
        self.out_scale = out_scale
        if resnet_type == 'resnet50':
            self.n_channels_out = 512
        elif resnet_type == 'resnet18':
            self.n_channels_out = 128

        max_out_scale = np.amax(self.out_scale)
        n_upscale = int(np.ceil(math.log(max_out_scale, 2)))
        self.scaler = nn.ModuleList([
            UNetUp(max(self.n_channels_out // (2**i), 64),
                   max(self.n_channels_out // (2**(i + 1)), 64),
                   bilinear=False,
                   norm=norm) for i in range(n_upscale)
        ])
        self.n_channels_out = max(self.n_channels_out // (2**(n_upscale)), 64)

    def forward(self, x):
        """
        Inputs:
            x - RGB image of size (bs, 3, H, W)
        """
        x_base = self.resnet_base(x)
        x_block1 = self.resnet_block1(x_base)
        x_block2 = self.resnet_block2(x_block1)
        x_feat = x_block2
        x_feat = F.adaptive_avg_pool2d(x_feat, 1)
        for mod in self.scaler:
            x_feat = mod(x_feat)
        return x_feat

    def train(self, mode=True):
        super().train(mode)
        for mod in self.resnet_base.modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                mod.eval()
        for mod in self.resnet_block1.modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                mod.eval()
        for mod in self.resnet_block2.modules():
            if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                mod.eval()

                