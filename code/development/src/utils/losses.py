"""
Implementation of various losses.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


# Code based on official tf implementation (github xuuuuuuchen/Active-Contour-Loss)
# and on unofficial pytorch implementation (github lc82111/Active-Contour-Loss-pytorch).
# Paper: https://ieeexplore.ieee.org/document/8953484
# Note: implementations have diverged from original paper, so equations might not match perfectly.
class ActiveContourLoss(nn.Module):
    def __init__(self, weight: float = 1.0, with_logits: bool = True) -> None:
        super().__init__()
        self.weight = weight
        self.with_logits = with_logits

    def forward(self, inputs: torch.tensor, targets: torch.tensor) -> float:
        if self.with_logits:
            inputs = torch.sigmoid(inputs)

        # Length term
        x = inputs[:, :, 1:, :] - inputs[:, :, :-1, :]  # horizontal and vertical directions
        y = inputs[:, :, :, 1:] - inputs[:, :, :, :-1]

        delta_x = x[:, :, 1:, :-2] ** 2
        delta_y = y[:, :, :-2, 1:] ** 2
        delta_u = torch.abs(delta_x + delta_y)

        eps = 1e-08
        length = torch.mean(torch.sqrt(delta_u + eps))  # eq. (11)

        # Region term
        C_1 = torch.ones_like(inputs)
        C_2 = torch.zeros_like(inputs)

        region_in = torch.abs(torch.mean(inputs * (C_1 - targets) ** 2))  # eq. (12)
        region_out = torch.abs(torch.mean((1 - inputs) * (C_2 - targets) ** 2))  # eq. (12)
        region = region_in + region_out

        return length + self.weight * region  # eq. (8)


class ActiveContourBCELoss(nn.Module):
    def __init__(self, with_logits: bool = True, w_bce: float = 0.7, w_acl: float = 0.3) -> None:
        super().__init__()
        self.with_logits = with_logits
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.acl = ActiveContourLoss(with_logits=self.with_logits)
        self.w_bce = w_bce
        self.w_acl = w_acl

    def forward(self, inputs: torch.tensor, targets: torch.tensor) -> float:
        bce = self.bce(inputs, targets)
        acl = self.acl(inputs, targets)

        return self.w_bce * bce + self.w_acl * acl


# Code based on official implementation: https://github.com/HiLab-git/ACELoss/blob/a9677a/aceloss.py#L89
class ACELoss(nn.Module):
    """Active Contour with Elastica (ACE) loss."""

    def __init__(
        self, u: float = 1.0, a: float = 1.0, b: float = 1.0, with_logits: bool = True, reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.u = u
        self.a = a
        self.b = b
        self.with_logits = with_logits
        self.reduction = reduction

    def first_derivative(self, input: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        u = input
        m = u.shape[2]
        n = u.shape[3]

        ci_0 = (u[:, :, 1, :] - u[:, :, 0, :]).unsqueeze(2)
        ci_1 = u[:, :, 2:, :] - u[:, :, 0 : m - 2, :]
        ci_2 = (u[:, :, -1, :] - u[:, :, m - 2, :]).unsqueeze(2)
        ci = torch.cat([ci_0, ci_1, ci_2], 2) / 2

        cj_0 = (u[:, :, :, 1] - u[:, :, :, 0]).unsqueeze(3)
        cj_1 = u[:, :, :, 2:] - u[:, :, :, 0 : n - 2]
        cj_2 = (u[:, :, :, -1] - u[:, :, :, n - 2]).unsqueeze(3)
        cj = torch.cat([cj_0, cj_1, cj_2], 3) / 2

        return ci, cj

    def second_derivative(
        self, input: torch.tensor, ci: torch.tensor, cj: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        u = input
        n = u.shape[3]

        cii_0 = (u[:, :, 1, :] + u[:, :, 0, :] - 2 * u[:, :, 0, :]).unsqueeze(2)
        cii_1 = u[:, :, 2:, :] + u[:, :, :-2, :] - 2 * u[:, :, 1:-1, :]
        cii_2 = (u[:, :, -1, :] + u[:, :, -2, :] - 2 * u[:, :, -1, :]).unsqueeze(2)
        cii = torch.cat([cii_0, cii_1, cii_2], 2)

        cjj_0 = (u[:, :, :, 1] + u[:, :, :, 0] - 2 * u[:, :, :, 0]).unsqueeze(3)
        cjj_1 = u[:, :, :, 2:] + u[:, :, :, :-2] - 2 * u[:, :, :, 1:-1]
        cjj_2 = (u[:, :, :, -1] + u[:, :, :, -2] - 2 * u[:, :, :, -1]).unsqueeze(3)

        cjj = torch.cat([cjj_0, cjj_1, cjj_2], 3)

        cij_0 = ci[:, :, :, 1:n]
        cij_1 = ci[:, :, :, -1].unsqueeze(3)

        cij_a = torch.cat([cij_0, cij_1], 3)
        cij_2 = ci[:, :, :, 0].unsqueeze(3)
        cij_3 = ci[:, :, :, 0 : n - 1]
        cij_b = torch.cat([cij_2, cij_3], 3)
        cij = cij_a - cij_b

        return cii, cjj, cij

    def region(self, y_pred: torch.tensor, y_true: torch.tensor, u: float = 1.0) -> torch.tensor:
        label = y_true.float()
        c_in = torch.ones_like(y_pred)
        c_out = torch.zeros_like(y_pred)
        if self.reduction == "mean":
            region_in = torch.abs(torch.mean(y_pred * ((label - c_in) ** 2)))
            region_out = torch.abs(torch.mean((1 - y_pred) * ((label - c_out) ** 2)))
        elif self.reduction == "sum":
            region_in = torch.abs(torch.sum(y_pred * ((label - c_in) ** 2)))
            region_out = torch.abs(torch.sum((1 - y_pred) * ((label - c_out) ** 2)))
        region = u * region_in + region_out
        return region

    def elastica(self, input: torch.tensor, a: float = 1.0, b: float = 1.0) -> torch.tensor:
        ci, cj = self.first_derivative(input)
        cii, cjj, cij = self.second_derivative(input, ci, cj)
        beta = 1e-8
        length = torch.sqrt(beta + ci ** 2 + cj ** 2)
        curvature = (beta + ci ** 2) * cjj + (beta + cj ** 2) * cii - 2 * ci * cj * cij
        curvature = torch.abs(curvature) / ((ci ** 2 + cj ** 2) ** 1.5 + beta)
        if self.reduction == "mean":
            elastica = torch.mean((a + b * (curvature ** 2)) * torch.abs(length))
        elif self.reduction == "sum":
            elastica = torch.sum((a + b * (curvature ** 2)) * torch.abs(length))
        return elastica

    def forward(self, inputs: torch.tensor, targets: torch.tensor) -> torch.tensor:
        if self.with_logits:
            inputs = torch.sigmoid(inputs)

        loss = self.region(inputs, targets, u=self.u) + self.elastica(inputs, a=self.a, b=self.b)
        return loss


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, batch_dice: bool = True, with_logits: bool = True) -> None:
        super().__init__()

        self.batch_dice = batch_dice
        self.smooth = smooth
        self.with_logits = with_logits

    def forward(self, inputs: torch.tensor, targets: torch.tensor) -> float:
        if self.with_logits:
            inputs = torch.sigmoid(inputs)

        if self.batch_dice:
            # flatten label and prediction tensors
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            soft_intersection = (inputs * targets).sum()
            dice = (2.0 * soft_intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        else:
            soft_intersection = (inputs * targets).sum(axis=[1, 2])
            dice = (2.0 * soft_intersection + self.smooth) / (
                inputs.sum(axis=[1, 2]) + targets.sum(axis=[1, 2]) + self.smooth
            )
            dice = dice.mean()

        return 1 - dice


class DiceACELoss(nn.Module):
    def __init__(
        self, batch_dice: bool = True, with_logits: bool = True, w_dice: float = 0.8, w_ace: float = 0.2
    ) -> None:
        super().__init__()
        self.batch_dice = batch_dice
        self.with_logits = with_logits
        self.dice = SoftDiceLoss(batch_dice=self.batch_dice, with_logits=self.with_logits)
        self.ace = ACELoss(with_logits=self.with_logits, reduction="mean")
        self.w_dice = w_dice
        self.w_ace = w_ace

    def forward(self, inputs: torch.tensor, targets: torch.tensor) -> float:
        dice = self.dice(inputs, targets)
        ace = self.ace(inputs, targets)

        return self.w_dice * dice + self.w_ace * ace


def contour(x: torch.tensor) -> torch.tensor:
    min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
    max_min_pool_x = torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1)
    contour = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
    return contour


class ContourSoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, with_logits: bool = True) -> None:
        super().__init__()

        self.smooth = smooth
        self.with_logits = with_logits

    def forward(self, inputs: torch.tensor, targets: torch.tensor) -> float:
        if self.with_logits:
            inputs = torch.sigmoid(inputs)

        weights = contour(targets) * 2
        weights = weights.view(-1)

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        soft_intersection = (inputs * (targets + weights)).sum()
        dice = (2.0 * soft_intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


def get_loss(loss: Optional[str] = None) -> Union[list, torch.Tensor]:
    implemented_losses = {
        # loss_name: (LossClass, out_channels)
        "acl": (ActiveContourLoss(), 1),
        "bce": (torch.nn.BCEWithLogitsLoss(), 1),
        "dice": (SoftDiceLoss(), 1),
        "ace": (ACELoss(), 1),
        "dice_ace": (DiceACELoss(), 1),
        "mse": (torch.nn.MSELoss(), 1),
        "cdl": (ContourSoftDiceLoss(), 1),
        "ac_bce": (ActiveContourBCELoss(), 1),
    }

    if loss not in implemented_losses:
        raise NotImplementedError(f"Loss {loss} is not available (must be: {', '.join(implemented_losses.keys())})")

    return implemented_losses[loss]
