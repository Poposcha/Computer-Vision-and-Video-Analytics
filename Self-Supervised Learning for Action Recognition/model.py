from typing import Any, List, Type, Union, Optional, Callable
import torch.nn as nn
from torch import Tensor
import torch
from torchvision.models import resnet18 as resnet18_2d


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv3d:
    """3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm3d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
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


class ResNet(nn.Module):
    def __init__(
        self,
        layers: List[int],
    ) -> None:
        super(ResNet, self).__init__()
        self.num_classes = 1000
        self._norm_layer = nn.BatchNorm3d

        self.inplanes = 64
        self.conv1 = nn.Conv3d(
            3, self.inplanes, kernel_size=7, stride=(1, 2, 2), padding=3, bias=False
        )
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, self.num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        # block = BasicBlock

        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(
            BasicBlock(
                self.inplanes,
                planes,
                stride,
                downsample,
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.inplanes,
                    planes,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(layers: List[int]) -> ResNet:
    model = ResNet(layers)
    return model


def resnet18(num_classes: int, inflated: bool = False) -> ResNet:
    m3 = _resnet([2, 2, 2, 2])
    m3.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    if inflated:
        # with torch.no_grad():
        m2 = resnet18_2d(pretrained=True)

        m2_state_dict = m2.state_dict()
        m3_state_dict = m3.state_dict()

        for n in m2_state_dict.keys():
            p2 = m2_state_dict[n]
            if "conv" in n:
                kernel_size = p2.shape[2]
                m3_state_dict[n].copy_(p2.unsqueeze(2).repeat(1, 1, kernel_size, 1, 1) / kernel_size)
            elif "bn" in n:
                m3_state_dict[n].copy_(p2)
        m3.load_state_dict(m3_state_dict)
    return m3
