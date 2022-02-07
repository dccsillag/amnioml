"""
Linear regression model for volume regression.
"""

from typing import Any

import torch
import torch.nn as nn

from src.utils.model_utils import CommonLightningModule


class Model(CommonLightningModule):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.save_hyperparameters()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            in_features=self.image_size ** 3, out_features=1, bias=True  # (height x width x depth) of 3D exam.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        x = x.reshape(-1)
        return x
