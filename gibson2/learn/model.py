# code adapted from https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F

from gibson2.learn.resnet import  resnet9w32gn_ws

class UNet(nn.Module):
    def __init__(self, input_channels=4, output_channels=2):
        super(UNet, self).__init__()
        self.backbone = resnet9w32gn_ws(input_channels)
        factor = 2
        layer_size = [256, 128, 64, 32, 32]
        self.up1 = Up(layer_size[0] + layer_size[1], layer_size[1])
        self.up2 = Up(layer_size[1] + layer_size[2], layer_size[2])
        self.up3 = Up(layer_size[2] + layer_size[3], layer_size[3])
        self.up4 = Up(layer_size[3] + layer_size[4], layer_size[4])
        self.readout = DoubleConv(layer_size[4], 
                                  output_channels, 
                                  mid_channels=layer_size[4] // 2)

    def forward(self, x):
        x1,x2,x3,x4,x5 = self.backbone(x)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        y = self.readout(x)
        return y,x


""" Parts of the U-Net model """

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.double_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

