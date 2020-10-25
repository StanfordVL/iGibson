import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as _models

class Conv2dWS(nn.Conv2d):
    """
    Weight Standardization
    """

    def forward(self, x):
        weight = self.weight
        weight_mean = (
            weight.mean(dim=1, keepdim=True)
            .mean(dim=2, keepdim=True)
            .mean(dim=3, keepdim=True)
        )
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

def create_encoder(model_name, **kwargs):
    return get_encoder_class(model_name)(**kwargs)

def _conv_op(ws):
    return Conv2dWS if ws else nn.Conv2d

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, ws=False):
    """
    3x3 convolution with padding
    ws: weight standardization
    """
    return _conv_op(ws)(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, ws=False):
    """1x1 convolution"""
    return _conv_op(ws)(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _get_nonlinearity(name: str):
    """
    Activation function
    """
    if name == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    elif name == "elu":
        return nn.ELU(inplace=True)
    elif name == "selu":
        return nn.SELU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        nonlinearity="relu",
        ws=False,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, ws=ws)
        self.bn1 = norm_layer(planes)
        self.relu = _get_nonlinearity(nonlinearity)
        self.conv2 = conv3x3(planes, planes, ws=ws)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    ''' 
    Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    while original implementation places the stride at the first 1x1 convolution(self.conv1)
    according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    This variant is also known as ResNet V1.5 and improves accuracy according to
    https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    '''

    expansion = 4
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        nonlinearity="relu",
        ws=False,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, ws=ws)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, ws=ws)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, ws=ws)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = _get_nonlinearity(nonlinearity)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        input_channels=4,
        base_width=64,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        nonlinearity="relu",
        ws=False,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = base_width
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.width_per_group = width_per_group
        self.ws = ws
        self.conv1 = _conv_op(ws)(
            input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.nonlinearity = nonlinearity
        self.relu = _get_nonlinearity(nonlinearity)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, base_width, layers[0])
        self.layer2 = self._make_layer(
            block,
            base_width * 2,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            base_width * 4,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            base_width * 8,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, ws=self.ws),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.width_per_group,
                previous_dilation,
                norm_layer=norm_layer,
                nonlinearity=self.nonlinearity,
                ws=self.ws,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.width_per_group,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    nonlinearity=self.nonlinearity,
                    ws=self.ws,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1,x2,x3,x4,x5

    def forward(self, x):
        return self._forward_impl(x)


# work around for pytorch DDP, cannot pickle lambda
class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_channels):
        super().__init__(num_groups=32, num_channels=num_channels, affine=True)


class GroupNorm16(nn.GroupNorm):
    def __init__(self, num_channels):
        super().__init__(num_groups=16, num_channels=num_channels, affine=True)


def _resnet_basic_gn(
    input_channels,
    num_layers, base_width, 
    nonlinearity="relu", *, ws=False
):
    if base_width == 64:
        gn_layer = GroupNorm32
    elif base_width == 32:
        gn_layer = GroupNorm16
    else:
        raise NotImplementedError

    layers = [1, 1, 1, 1]
    return ResNet(
        BasicBlock,
        layers,
        input_channels=input_channels,
        base_width=base_width,
        norm_layer=gn_layer,
        nonlinearity=nonlinearity,
        ws=ws,
    )


def resnet9w32(pretrained=False):
    return ResNet(
        BasicBlock,
        [1, 1, 1, 1],
        base_width=32,
        nonlinearity="relu",
        ws=False,
    )

def resnet9w32gn_ws(input_channels=4, pretrained=False):
    return _resnet_basic_gn(input_channels, 9, 32, "relu", ws=True)


