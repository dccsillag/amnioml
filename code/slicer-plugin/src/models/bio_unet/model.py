import argparse
from typing import Any, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
#from src.utils.model_utils import CommonLightningModule

class ModelPrediction2d:
    """Manipulate and save predictions (in 3D), for models that use HdfDataset2d.

    Given the model, the dataloader and the path to an hdf5 dataset, this class
    provides a dictionary-like object to access the realigned 3D predictions.

    On-the-fly predictions are accessible via bb[<subject_id>], and just use
    bb.save(<path_to_hdf5>) to save.

    Example:

    INIT
        model = UNet.load_from_checkpoint("<checkpoint>.ckpt")
        model.to("cuda")
        dataloader = src.models.bio_unet.dataloader.DataLoader(<dataloader_args>)
        bb = BlackBox2d(model, dataloader, "<hdf_dataset_2d>.h5", original_resolution=True)

    PREDICT ON-THE-FLY
        bb.keys() # all subject-ids available in the dataset
        bb['<str-subject-id>'] # the prediction for subject, already aligned in 3D

    SAVE
        bb.save("<pred>.hdf5") # save all predictions to an hdf file
        bb.save("<pred>.hdf5", subject_ids) # save predictions of subject-ids the list to an hdf file

    READ SAVED FILES
        f = h5py.File("<pred>.hdf5", "r")
        f.keys() # list of available subject-ids
        f['<subject_id>'] # prediction (in 3D) for <subject_id>
        np.arary(f['<subject_id>']) # np.array for <subject_id>

    """

    def __init__(
        self,
        pre_trained_model: Any,
        dataloader: Any,
        dataset_path: str,
        sigmoid: bool = True,
        original_resolution: bool = False,
    ) -> None:
        self.sigmoid = sigmoid
        self.model = pre_trained_model
        self.model.freeze()
        self.model.eval()
        self.original_resolution = original_resolution
        torch_dataloader, self.dataset_ids = dataloader.get_dataloader_and_dataset_ids(
            dataset_path, batch_size=1, shuffle=False
        )
        self.dataset = torch_dataloader.dataset
        self.subject_id_2_slices: Dict[str, Dict] = {}
        for i in range(len(self.dataset_ids)):
            subject_id, slice_i = self.dataset_ids[i]
            self.subject_id_2_slices.setdefault(subject_id, {})
            self.subject_id_2_slices[subject_id][int(slice_i)] = i

        self.id_2_pool_paths = get_id_2_pool_paths(list(self.subject_id_2_slices.keys()))

    def keys(self):
        return self.subject_id_2_slices.keys()

    def _get_slice(self, dataset_index: int, size: Any = [-1, -1]):
        with torch.no_grad():
            s = self.model(self.dataset[dataset_index][0].unsqueeze(0).to(self.model.device.type)).to("cpu")
        if self.sigmoid:
            s = torch.sigmoid(s)
        if size[0] > 0 and size[1] > 0 and (size[0] != s.shape[2] or size[1] != s.shape[3]):
            s = torchvision.transforms.Resize((size[0], size[1]), interpolation=InterpolationMode.BILINEAR)(s)
        return s.squeeze().squeeze().detach().numpy()

    def _get_shape(self, subject_id: str):
        slice_2_index = self.subject_id_2_slices[subject_id]
        nrrd_header = nrrd.read_header(self.id_2_pool_paths[subject_id][0])
        shape = nrrd_header["sizes"]
        if not self.original_resolution:
            shape_seg = self.dataset[next(iter(slice_2_index))][1].shape
            shape[:2] = shape_seg[1:]
        return shape

    def __getitem__(self, subject_id: str):
        slice_2_index = self.subject_id_2_slices[subject_id]
        shape = self._get_shape(subject_id)
        pred = np.zeros(self._get_shape(subject_id), dtype=np.float32)
        iter_slices = list(slice_2_index.keys())
        for j in range(len(iter_slices)):
            i = iter_slices[j]
            pred[:, :, i] = self._get_slice(slice_2_index[i], shape[:2])
        return pred

    def get_slice(self, subject_id: str, slice_index: int):
        i = self.subject_id_2_slices[subject_id][slice_index]
        shape = self._get_shape(subject_id)
        return self._get_slice(i, shape[:2])

    def save(self, path_to_hdf: str, subject_ids = []) -> None:
        if os.path.exists(path_to_hdf):
            raise FileExistsError(f"{path_to_hdf} already exists.")
        f = h5py.File((path_to_hdf), "w")

        if len(subject_ids) >= 1:
            list_of_subject_ids = subject_ids
        else:
            list_of_subject_ids = list(self.subject_id_2_slices.keys())

        for i in trange(len(list_of_subject_ids)):
            subject_id = list_of_subject_ids[i]
            f.create_dataset(subject_id, data=self[subject_id], **hdf5plugin.Blosc(), track_times=False)
        f.close()


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
    def __init__(
        self, with_logits: bool = True, w_bce: float = 0.7, w_acl: float = 0.3
    ) -> None:
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

    def first_derivative(self, input: torch.tensor):
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
    ):
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


def contour(x):
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


def get_loss(loss: Optional[str] = None):
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

def get_loss(loss: Optional[str] = None):
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

class CommonLightningModule(pl.LightningModule):
    def __init__(
        self, *, loss: str, optimizer: str, learning_rate: float, image_size: int, n_channels: int, args
    ):
        super().__init__()

        self.loss_name = loss
        self.loss, _ = get_loss(self.loss_name)
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.n_channels = n_channels
        self.n_classes = 1
        self.args = args

    def _eval_step_results(self, prefix: str, out: torch.Tensor, target: torch.Tensor, loss: float) -> float:
        self.log(prefix + "_loss", loss, sync_dist=True, on_epoch=True, on_step=False)
        if prefix != "train" and self.loss_name != "mse":
            self.log(
                prefix + "_bce", torch.nn.BCEWithLogitsLoss()(out, target), sync_dist=True, on_epoch=True, on_step=False
            )
            self.log(prefix + "_dice", SoftDiceLoss()(out, target), sync_dist=True, on_epoch=True, on_step=False)

        return loss

    def training_epoch_end(self, outs):
        # This makes us log training metrics with the X axis being the epoch number:
        self.log("step", self.trainer.current_epoch)
        return super().training_epoch_end(outs)

    def validation_epoch_end(self, outs):
        # This makes us log validation metrics with the X axis being the epoch number:
        self.log("step", self.trainer.current_epoch)
        return super().validation_epoch_end(outs)

    def training_step(self, train_batch: Any, batch_idx: int) -> float:
        input, target = train_batch
        out = self(input)
        loss = self.loss(out, target)
        return self._eval_step_results("train", out, target, loss)

    def validation_step(self, val_batch: Any, batch_idx: int) -> float:
        input, target = val_batch
        out = self(input)
        loss = self.loss(out, target)
        return self._eval_step_results("val", out, target, loss)

    def test_step(self, test_batch: Any, batch_idx: int) -> float:
        input, target = test_batch
        out = self(input)
        loss = self.loss(out, target)
        return self._eval_step_results("test", out, target, loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "rmsprop":
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=1e-8, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer in CommonLightningModule: {self.optimizer}")

    def predict(self, dataset_path: str) -> ModelPrediction2d:
        dataloader = HdfDataloader2d(
            "",
            num_workers=1,
            batch_size=1,
            image_size=int(self.hparams["args"]["image_size"]),
            n_slices=int(self.hparams["args"]["num_channels"]),
        )
        return ModelPrediction2d(self, dataloader, dataset_path)



def add_predefined_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bilinear", action="store_true", help="Whether to use bilinear resizing")


class Model(CommonLightningModule):
    def __init__(self, bilinear: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.bilinear = bilinear

        self.save_hyperparameters()

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
