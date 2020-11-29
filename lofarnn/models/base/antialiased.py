import torch.nn as nn
from antialiased_cnns import BlurPool

"""
Antialiased ResNet
"""


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups,
        bias=False,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        norm_layer=None,
        filter_size=1,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError("BasicBlock only supports groups=1")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1:
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = nn.Sequential(
                BlurPool(planes, filt_size=filter_size, stride=stride),
                conv3x3(planes, planes),
            )
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
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        norm_layer=None,
        filter_size=1,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, groups)  # stride moved
        self.bn2 = norm_layer(planes)
        if stride == 1:
            self.conv3 = conv1x1(planes, planes * self.expansion)
        else:
            self.conv3 = nn.Sequential(
                BlurPool(planes, filt_size=filter_size, stride=stride),
                conv1x1(planes, planes * self.expansion),
            )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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


class AntiAliasedResNet(nn.Module):
    def __init__(
        self,
        num_channels,
        block,
        layers,
        num_classes=2,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        norm_layer=None,
        filter_size=1,
        pool_only=True,
        keep_fc=True,
    ):
        super(AntiAliasedResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.inplanes = 64
        self.keep_fc = keep_fc

        if pool_only:
            self.conv1 = nn.Conv2d(
                num_channels,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        else:
            self.conv1 = nn.Conv2d(
                num_channels,
                self.inplanes,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if pool_only:
            self.maxpool = nn.Sequential(
                *[
                    nn.MaxPool2d(kernel_size=2, stride=1),
                    BlurPool(planes[0], filt_size=filter_size, stride=2,),
                ]
            )
        else:
            self.maxpool = nn.Sequential(
                *[
                    BlurPool(planes[0], filt_size=filter_size, stride=2,),
                    nn.MaxPool2d(kernel_size=2, stride=1),
                    BlurPool(planes[0], filt_size=filter_size, stride=2,),
                ]
            )

        self.layer1 = self._make_layer(
            block, 64, layers[0], groups=groups, norm_layer=norm_layer
        )
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            groups=groups,
            norm_layer=norm_layer,
            filter_size=filter_size,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            groups=groups,
            norm_layer=norm_layer,
            filter_size=filter_size,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            groups=groups,
            norm_layer=norm_layer,
            filter_size=filter_size,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if (
                    m.in_channels != m.out_channels
                    or m.out_channels != m.groups
                    or m.bias is not None
                ):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                else:
                    print("Not initializing")
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

    def _make_layer(
        self, block, planes, blocks, stride=1, groups=1, norm_layer=None, filter_size=1
    ):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride, filter_size=filter_size),
            #     norm_layer(planes * block.expansion),
            # )

            downsample = (
                [
                    BlurPool(
                        filt_size=filter_size, stride=stride, channels=self.inplanes
                    ),
                ]
                if (stride != 1)
                else []
            )
            downsample += [
                conv1x1(self.inplanes, planes * block.expansion, 1),
                norm_layer(planes * block.expansion),
            ]
            # print(downsample)
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                groups,
                norm_layer,
                filter_size=filter_size,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=groups,
                    norm_layer=norm_layer,
                    filter_size=filter_size,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.keep_fc:
            x = self.fc(x)

        return x
