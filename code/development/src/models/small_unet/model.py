"""
Small U-Net model.

Adapted from https://github.com/shreyaspadhy/UNet-Zoo/blob/e2b8f38/models.py
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from src.utils.model_utils import CommonLightningModule, Upsample


class Model(CommonLightningModule):
    NAME = "Small U-Net"

    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__(**kwargs)
        num_feat = [32, 64, 128, 256]

        self.save_hyperparameters()

        self.down1 = nn.Sequential(Conv3x3Small(self.n_channels, num_feat[0]))

        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(num_feat[0]), Conv3x3Small(num_feat[0], num_feat[1])
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(num_feat[1]), Conv3x3Small(num_feat[1], num_feat[2])
        )

        self.bottom = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(num_feat[2]),
            Conv3x3Small(num_feat[2], num_feat[3]),
            nn.BatchNorm2d(num_feat[3]),
        )

        self.up1 = UpSample(num_feat[3], num_feat[2])
        self.upconv1 = nn.Sequential(Conv3x3Small(num_feat[3] + num_feat[2], num_feat[2]), nn.BatchNorm2d(num_feat[2]))

        self.up2 = UpSample(num_feat[2], num_feat[1])
        self.upconv2 = nn.Sequential(Conv3x3Small(num_feat[2] + num_feat[1], num_feat[1]), nn.BatchNorm2d(num_feat[1]))

        self.up3 = UpSample(num_feat[1], num_feat[0])
        self.upconv3 = nn.Sequential(Conv3x3Small(num_feat[1] + num_feat[0], num_feat[0]), nn.BatchNorm2d(num_feat[0]))

        self.final = nn.Sequential(nn.Conv2d(num_feat[0], 1, kernel_size=1))

    def forward(self, inputs: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        down1_feat = self.down1(inputs)
        down2_feat = self.down2(down1_feat)
        down3_feat = self.down3(down2_feat)
        bottom_feat = self.bottom(down3_feat)

        up1_feat = self.up1(bottom_feat, down3_feat)
        up1_feat = self.upconv1(up1_feat)
        up2_feat = self.up2(up1_feat, down2_feat)
        up2_feat = self.upconv2(up2_feat)
        up3_feat = self.up3(up2_feat, down1_feat)
        up3_feat = self.upconv3(up3_feat)

        if return_features:
            outputs = up3_feat
        else:
            outputs = self.final(up3_feat)

        return outputs


class Conv3x3Small(nn.Module):
    def __init__(self, in_feat: torch.Tensor, out_feat: torch.Tensor) -> None:
        super(Conv3x3Small, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1), nn.ELU(inplace=True), nn.Dropout(p=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1), nn.ELU(inplace=True)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UpSample(nn.Module):
    def __init__(self, in_feat: torch.Tensor, out_feat: torch.Tensor) -> None:
        super(UpSample, self).__init__()

        self.up = Upsample(scale_factor=2)

        self.deconv = nn.ConvTranspose2d(in_feat, out_feat, kernel_size=2, stride=2)

    def forward(self, inputs: torch.Tensor, down_outputs: torch.Tensor) -> torch.Tensor:
        outputs = self.up(inputs)
        out = torch.cat([outputs, down_outputs], 1)
        return out
