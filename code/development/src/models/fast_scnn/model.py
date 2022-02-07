"""
Fast Semantic Segmentation Network (Fast-SCNN). Part of this custom
implementation is inspired by and makes use of the following github
repositories: Tramac/Fast-SCNN-pytorch and bernardomig/ark.
"""

from typing import Any, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from src.utils.model_utils import CommonLightningModule


class Model(CommonLightningModule):
    NAME = "Fast-SCNN"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.downsample = nn.Sequential(
            ConvBnReLU2d(self.n_channels, 32, 3, stride=2, padding=1),
            DSConv(32, 48, stride=2),
            DSConv(48, 64, stride=2),
        )

        self.extraction = nn.Sequential(
            BottleneckModule(64, 64, stride=2, expansion=6),
            BottleneckModule(64, 96, stride=2, expansion=6),
            BottleneckModule(96, 128, stride=1, expansion=6),
            PyramidPoolingModule(128, 128),
        )

        self.fusion = FeatureFusionModule((128, 64), 128, scale_factor=4)

        self.classification = nn.Sequential(
            DSConv(128, 128, stride=1),
            DSConv(128, 128, stride=1),
            nn.Dropout(0.1),
            nn.Conv2d(128, self.n_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        downsample = self.downsample(x)
        extraction = self.extraction(downsample)
        fusion = self.fusion(extraction, downsample)
        classes = self.classification(fusion)

        return F.interpolate(classes, size=x.shape[2:], mode="bilinear", align_corners=True)


class BottleneckModule(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: Union[int, Tuple], expansion: int) -> None:
        super().__init__()
        self.bottleneck1 = BottleneckBlock(in_channels, out_channels, stride=stride, expansion=expansion)
        self.bottleneck2 = BottleneckBlock(out_channels, out_channels, stride=1, expansion=expansion)
        self.bottleneck3 = BottleneckBlock(out_channels, out_channels, stride=1, expansion=expansion)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: Union[int, Tuple], expansion: int) -> None:
        super().__init__()
        expansion_channels = expansion * in_channels
        self.conv1 = ConvBnReLU2d(in_channels, expansion_channels, 1, stride=1)
        self.conv2 = ConvBnReLU2d(
            expansion_channels, expansion_channels, 3, stride=stride, padding=1, groups=expansion_channels
        )
        self.conv3 = ConvBn2d(expansion_channels, out_channels, 1, stride=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)

        if x.shape == input.shape:
            x += input

        return F.relu(x)


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_scales: Tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        inter_channels = in_channels // len(pool_scales)
        self.pyramid = [
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvBnReLU2d(in_channels, inter_channels, 1),
            )
            for scale in pool_scales
        ]
        self.pyramid = nn.ModuleList(self.pyramid)
        self.out = ConvBnReLU2d(in_channels * 2, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.size()[2:]
        x = [x] + [F.interpolate(level(x), size, mode="bilinear", align_corners=True) for level in self.pyramid]
        x = torch.cat(x, dim=1)
        x = self.out(x)
        return x


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels: Tuple[int, int], out_channels: int, scale_factor: Union[int, Tuple]) -> None:
        super().__init__()
        lowres_channels, highres_channels = in_channels

        self.lowres_conv = nn.Sequential(
            ConvBnReLU2d(
                lowres_channels,
                lowres_channels,
                3,
                stride=1,
                padding=scale_factor,
                dilation=scale_factor,
                groups=lowres_channels,
            ),
            ConvBn2d(lowres_channels, out_channels, 1),
        )

        self.highres_conv = ConvBn2d(highres_channels, out_channels, 1)

    def forward(self, lowres: torch.Tensor, highres: torch.Tensor) -> torch.Tensor:
        lowres = F.interpolate(lowres, size=highres.shape[2:], mode="bilinear", align_corners=True)
        lowres = self.lowres_conv(lowres)
        highres = self.highres_conv(highres)
        return F.relu(lowres + highres)


class DSConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: Union[int, Tuple]) -> None:
        super().__init__()
        self.depthwise_conv = ConvBn2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels)
        self.pointwise_conv = ConvBnReLU2d(in_channels, out_channels, 1)


class ConvBn2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple] = 0,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)


class ConvBnReLU2d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple] = 0,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv_bn = ConvBn2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.relu = nn.ReLU(inplace=True)
