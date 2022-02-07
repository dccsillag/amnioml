"""
Utils for the models pipeline.
"""

import argparse
import datetime
import importlib
import os
import random
from glob import glob
from typing import Any, Callable, Dict, KeysView, List, Optional, Tuple

import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import InterpolationMode
from tqdm import tqdm, trange

from src.utils.data_utils import VolumeRegressionDataset, get_normalized_exam_slice
from src.utils.general_utils import normalize_image, outpath
from src.utils.losses import SoftDiceLoss, get_loss


def get_model_module(model_name: str) -> Any:
    return importlib.import_module(f"src.models.{model_name}.model")


def get_model_class(model_name: str) -> "CommonLightningModule":
    return get_model_module(model_name).Model


def get_model_name(model_name: str) -> str:
    return get_model_class(model_name).NAME


def get_model_args(args: List, add_predefined_args: Callable[[argparse.ArgumentParser], None]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_predefined_args(parser)
    return parser.parse_args(args)


class HdfDataset2d(torch.utils.data.Dataset):
    """Data loader for the hdf_db format.

    self.n_slices is the number of channels of the exam per object.
    self.stride is the inner stride/step of the channels (see the graphical representation bellow).

    A graphical representation of the slices is shown bellow:

            i             Exam                Segmentation

        [i-stride]    ------------
        [i       ]    ------------    --->    ------------
        [i+stride]    ------------

    Note that this pattern is repeated for all possible values of i.

    Example: with n_slices=3, we would have

        dataset = HdfDataset2d(<path to .h5>, <exam_transforms>, <segmentation_transforms>, n_slices=3)
        dataset[k].shape = [3, x, y]
        dataset[k].shape = [x, y]

    Note: some segmentations have more than 2 values, so normalizing by max or
    ptp would have undesirable results. Hence, the following convention was adopted:

    Normalize the segmentation with a nonzero test.
    """

    def __init__(
        self,
        dataset_path: str,
        exam_transforms: List[Callable[[torch.Tensor], torch.Tensor]],
        segmentation_transforms: List[Callable[[torch.Tensor], torch.Tensor]],
        n_slices: int = 1,
        stride: int = 1,
        normalize_using_center: bool = False,
    ) -> None:
        self.dataset_path = dataset_path
        self.exam_transforms = exam_transforms
        self.segmentation_transforms = segmentation_transforms
        self.inputs_dtype = torch.float32
        self.outputs_dtype = torch.float32
        if n_slices % 2 != 1:
            raise ValueError(f"n_slices needs to be an odd number. n_slices={n_slices} is even.")
        self.half_n_slices = int((n_slices - 1) / 2)
        if stride <= 0:
            raise ValueError(f"stride={stride} <= 0.")
        self.stride = stride
        self.f = h5py.File(dataset_path, "r")
        list_of_subject_ids = list(self.f.keys())
        self.normalize_using_center = normalize_using_center

        # list of [(subject_id, slice_i)]
        self.ids_with_slices = []
        for subject_id in list_of_subject_ids:
            total_number_of_slices = len(self.f[subject_id]["slices"].keys())
            for slice_i in self.f[subject_id]["slices"].keys():
                # Only include slices that fall inside bounds
                if (
                    0 <= int(slice_i) - self.half_n_slices * self.stride
                    and int(slice_i) + self.half_n_slices * self.stride < total_number_of_slices
                ):
                    self.ids_with_slices.append((subject_id, slice_i))

    def __len__(self) -> int:
        return len(self.ids_with_slices)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        subject_id, slice_i = self.ids_with_slices[i]

        # Create an array of slices, whose indexes are in the interval [-self.half_n_slices, +self.half_n_slices]
        if self.normalize_using_center:
            exam = np.array(
                [
                    get_normalized_exam_slice(
                        self.f[subject_id]["slices"][str(int(slice_i) + self.stride * (j - self.half_n_slices))]["exam"]
                    )
                    for j in range(2 * self.half_n_slices + 1)
                ],
                dtype=np.float32,
            )
        else:
            exam = np.array(
                [
                    self.f[subject_id]["slices"][str(int(slice_i) + self.stride * (j - self.half_n_slices))]["exam"]
                    for j in range(2 * self.half_n_slices + 1)
                ],
                dtype=np.float32,
            )
        segmentation = np.array(self.f[subject_id]["slices"][slice_i]["segmentation"])

        # Add extra dimension to segmentation and exam
        segmentation = np.expand_dims(segmentation, axis=2)

        # Normalize to 0-1
        if not self.normalize_using_center:
            exam = exam / max(exam.max(), 1.0)
        segmentation = np.array(segmentation != 0, dtype=np.float32)

        # Convert to torch tensor
        exam = torch.from_numpy(np.array(exam))
        segmentation = torch.from_numpy(np.array(segmentation))

        # Change order to [depth, height, width]
        segmentation = segmentation.permute(2, 0, 1)

        # Apply exam transforms
        if self.exam_transforms is not None:
            for transform in self.exam_transforms:
                exam = transform(exam)

        # Apply segmentation transforms
        if self.segmentation_transforms is not None:
            for transform in self.segmentation_transforms:
                segmentation = transform(segmentation)

        # Typecasting
        exam = exam.to(self.inputs_dtype)
        segmentation = segmentation.to(self.outputs_dtype)

        return exam, segmentation


class HdfDataloader2d(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        image_size: int,
        num_workers: int,
        batch_size: int,
        n_slices: int,
        stride: int = 1,
        normalize_using_center: bool = False,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.n_slices = n_slices
        self.stride = stride
        self.normalize_using_center = normalize_using_center
        self.exam_transforms = [
            torchvision.transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
        ]
        self.segmentation_transforms = [
            torchvision.transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)
        ]
        self.train_dataset_path = f"data/processed/{self.dataset}/train.hdf"
        self.val_dataset_path = f"data/processed/{self.dataset}/val.hdf"
        self.test_dataset_path = f"data/processed/{self.dataset}/test.hdf"

    def get_dataloader_and_dataset_ids(
        self, dataset_path: str, batch_size: int, shuffle: bool = False
    ) -> Tuple[torch.utils.data.DataLoader, List[Tuple[str, str]]]:
        dataset = HdfDataset2d(
            dataset_path=dataset_path,
            exam_transforms=self.exam_transforms,
            segmentation_transforms=self.segmentation_transforms,
            n_slices=self.n_slices,
            stride=self.stride,
            normalize_using_center=self.normalize_using_center,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        return dataloader, dataset.ids_with_slices

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader_and_dataset_ids(self.train_dataset_path, self.batch_size, shuffle=True)[0]

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader_and_dataset_ids(self.val_dataset_path, 4, shuffle=False)[0]

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader_and_dataset_ids(self.test_dataset_path, 4, shuffle=False)[0]


class DataloaderVolumeRegression(pl.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        image_size: int,
        num_workers: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.exam_transforms = [
            lambda x: F.interpolate(x.unsqueeze(0).unsqueeze(0), (image_size, image_size, image_size))[0, 0, ...]
        ]
        self.segmentation_transforms: List[Any] = []
        self.train_dataset_path = f"data/{self.dataset}/train"
        self.val_dataset_path = f"data/{self.dataset}/val"
        self.test_dataset_path = f"data/{self.dataset}/test"

    def get_dataloader_and_dataset_ids(
        self, dataset_path: str, batch_size: int, shuffle: bool = False
    ) -> Tuple[torch.utils.data.DataLoader, None]:
        dataset = VolumeRegressionDataset(
            subject_folders=sorted(glob(os.path.join(dataset_path, "*.hdf5"))),
            exam_transforms=self.exam_transforms,
            segmentation_transforms=self.segmentation_transforms,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )
        return dataloader, None  # dataset.ids

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader_and_dataset_ids(self.train_dataset_path, self.batch_size, shuffle=True)[0]

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader_and_dataset_ids(self.val_dataset_path, 4, shuffle=False)[0]

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.get_dataloader_and_dataset_ids(self.test_dataset_path, 4, shuffle=False)[0]


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        args: argparse.Namespace,
        training_DataLoader: torch.utils.data.Dataset,
        validation_DataLoader: torch.utils.data.Dataset,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        criterion_name: str = "cross_entropy",
        epochs: int = 100,
        first_epoch: int = 0,
        output_folder: Optional[str] = None,
        save_checkpoint: bool = True,
        checkpoint_interval: int = 5,
    ):

        self.model = model
        self.criterion = criterion
        self.criterion_name = criterion_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.args = args
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.current_epoch = first_epoch
        self.output_folder = output_folder

        self.training_loss: List[float] = []
        self.validation_loss: List[float] = []
        self.learning_rate: List[float] = []
        self.best_val_loss: float = np.inf

        self.save_checkpoint = save_checkpoint
        self.checkpoint_interval = checkpoint_interval

    def fit(self) -> Tuple[List[float], List[float], List[float]]:

        progressbar = trange(self.epochs, desc="Progress")
        for i in progressbar:
            """Epoch counter"""
            self.current_epoch += 1  # epoch counter

            """Training block"""
            self._train()

            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()

                if self.validation_loss[-1] < self.best_val_loss:
                    self.best_val_loss = self.validation_loss[-1]
                    self._save_best_model()

            """Save checkpoint"""
            if self.save_checkpoint and self.current_epoch % self.checkpoint_interval == 0:
                self._save_checkpoint()

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if (
                    self.validation_DataLoader is not None
                    and self.lr_scheduler.__class__.__name__ == "ReduceLROnPlateau"
                ):
                    # learning rate scheduler step with validation loss
                    self.lr_scheduler.batch(self.validation_loss[i])
                else:
                    self.lr_scheduler.step()  # learning rate scheduler step
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self) -> None:

        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(
            enumerate(self.training_DataLoader),
            "Training",
            total=len(self.training_DataLoader),
            leave=False,
        )

        for i, (x, y) in batch_iter:
            input, target = (
                x.to(self.device),
                y.to(self.device),
            )  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            if isinstance(out, tuple):
                out = out[0]
            loss = self.criterion(out, target)  # calculate loss
            loss_value = loss.item()
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters

            batch_iter.set_description(f"Training: (loss {loss_value:.4f})")  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.learning_rate.append(self.optimizer.param_groups[0]["lr"])

        batch_iter.close()

    def _validate(self) -> None:

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(
            enumerate(self.validation_DataLoader),
            "Validation",
            total=len(self.validation_DataLoader),
            leave=False,
        )

        for i, (x, y) in batch_iter:
            input, target = (
                x.to(self.device),
                y.to(self.device),
            )  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                if isinstance(out, tuple):
                    out = out[0]
                loss = self.criterion(out, target)
                loss_value = loss.item()
                valid_losses.append(loss_value)

                batch_iter.set_description(f"Validation: (loss {loss_value:.4f})")

        self.validation_loss.append(np.mean(valid_losses))

        batch_iter.close()

    def _save_checkpoint(self) -> None:
        filename = f"checkpoint-{self.args.dataset}-{self.current_epoch}_{self.epochs}-{self.criterion_name}.pt"
        checkpoint_dict = {
            "epoch": self.current_epoch,
            "state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(checkpoint_dict, outpath(f"{self.output_folder}/checkpoints/{filename}"))

    def _save_best_model(self) -> None:
        filename = f"{self.output_folder}/model_best-{self.args.dataset}-{self.epochs}-{self.criterion_name}.pt"
        best_model_dict = {
            "epoch": self.current_epoch,
            "state_dict": self.model.state_dict(),
            "best_val_loss": self.best_val_loss,
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(best_model_dict, outpath(filename))

    def get_state(self) -> Dict:
        state = {
            "user": os.getlogin(),
            "time": str(datetime.datetime.now()),
            "model_state": self.model.state_dict(),
            "model_name": type(self.model).__name__,
            "optimizer_state": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            "loss": type(self.criterion).__name__,
            "device": str(self.device),
            "epochs": self.epochs,
            "lr_schedule": self.lr_scheduler,
            "args": {arg: getattr(self.args, arg) for arg in vars(self.args)},
            "train_loss": self.training_loss,
            "val_loss": self.validation_loss,
            "learning_rate": self.learning_rate,
        }
        return state


def plot_losses(
    training_losses: List[float],
    validation_losses: List[float],
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Returns a loss plot with training loss, validation loss and learning rate.
    """

    list_len = len(training_losses)
    x_range = list(range(1, list_len + 1))  # number of x values

    fig = plt.figure(figsize=figsize)
    ax = plt.axes()
    linestyle_original = "-"
    color_original_train = "orange"
    color_original_valid = "red"
    alpha = 1.0

    ax.plot(
        x_range,
        training_losses,
        linestyle_original,
        color=color_original_train,
        label="Training",
        alpha=alpha,
    )
    ax.plot(
        x_range,
        validation_losses,
        linestyle_original,
        color=color_original_valid,
        label="Validation",
        alpha=alpha,
    )
    ax.title.set_text("Training & validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.legend(loc="upper right")
    plt.close()

    return fig


class CommonLightningModule(pl.LightningModule):
    def __init__(
        self, *, loss: str, optimizer: str, learning_rate: float, image_size: int, n_channels: int, args: Dict[str, Any]
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

    def training_epoch_end(self, outs: List[Dict[str, float]]) -> None:
        # This makes us log training metrics with the X axis being the epoch number:
        self.log("step", self.trainer.current_epoch)
        return super().training_epoch_end(outs)

    def validation_epoch_end(self, outs: List[Dict[str, float]]) -> None:
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


class TensorBoardLogger(pl.Callback):
    def __init__(self, regression: bool = False):
        self.regression = regression

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: CommonLightningModule, _: Any) -> None:
        assert isinstance(pl_module, CommonLightningModule)

        self._log_stuff(trainer, pl_module, "train", trainer.train_dataloader)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: CommonLightningModule) -> None:
        assert isinstance(pl_module, CommonLightningModule)

        if trainer.running_sanity_check:
            return

        self._log_stuff(trainer, pl_module, "val", trainer.val_dataloaders[0])

    def _log_stuff(
        self,
        trainer: pl.Trainer,
        pl_module: CommonLightningModule,
        prefix: str,
        dataloader: torch.utils.data.DataLoader,
    ) -> None:
        assert isinstance(pl_module, CommonLightningModule)

        tensorboard = pl_module.logger.experiment

        pl_module.eval()

        with torch.no_grad():
            if prefix == "train":
                sample_batch = next(iter(dataloader))
                sample_batch = trainer.accelerator.batch_to_device(sample_batch)
                sample_input, sample_target = sample_batch
            else:
                sample_data = dataloader.dataset[random.randint(0, len(dataloader.dataset) - 1)]
                sample_data = trainer.accelerator.batch_to_device(sample_data)
                sample_input, sample_target = sample_data
                sample_input = sample_input[None, ...]
                sample_target = sample_target[None, ...]

            predictions = pl_module(sample_input)
            if not self.regression:
                predictions = torch.where(pl_module(sample_input) > 0.5, 1, 0)
                predictions = predictions[:, 0, ...]
                predictions = predictions.to("cpu").numpy()
                predictions = predictions[0, ...][None, ...]

                tensorboard.add_image(
                    prefix + "_sample_pred", normalize_image(predictions), pl_module.trainer.current_epoch
                )
                tensorboard.add_image(prefix + "_sample_target", sample_target[0, ...], pl_module.trainer.current_epoch)

        pl_module.train()


class Upsample(nn.Module):
    """Upsample input with shape [batch_size, channels, height, width] via nearest neighbors method. This is
    equivalent to nn.Upsample(scale_factor=2, mode='nearest') but much faster in TPUs. It works in three steps:
        1) add extra dimensions so input becomes [batch_size, channels, height, 1, ..., 1, width, 1, ..., 1];
        2) duplicate height and width information to new axes: [batch_size, channels, height, ..., height,
    width, ..., width]
        3) reshape so final shape is [batch_size, channels, scale_factor * height, scale_factor * width].
    """

    def __init__(self, scale_factor: int = 2) -> None:
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        batch_with_extra_dims = batch[:, :, :, None, :, None]
        batch_with_extra_dims = batch_with_extra_dims.expand(-1, -1, -1, self.scale_factor, -1, self.scale_factor)
        output = batch_with_extra_dims.reshape(
            batch.size(0), batch.size(1), self.scale_factor * batch.size(2), self.scale_factor * batch.size(3)
        )
        return output


smp_models = [
    # See https://github.com/qubvel/segmentation_models.pytorch/tree/4f94380815f831605f4641b7193df2eccd5652a3#architectures-
    "Unet",
    "UnetPlusPlus",
    "MAnet",
    "Linknet",
    "FPN",
    "PSPNet",
    "DeepLabV3",
    "DeepLabV3Plus",
    "PAN",
]
smp_encoders = [
    # See https://github.com/qubvel/segmentation_models.pytorch/tree/4f94380815f831605f4641b7193df2eccd5652a3#encoders-
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "timm-resnest14d",
    "timm-resnest26d",
    "timm-resnest50d",
    "timm-resnest101e",
    "timm-resnest200e",
    "timm-resnest269e",
    "timm-resnest50d_4s2x40d",
    "timm-resnest50d_1s4x24d",
    "timm-res2net50_26w_4s",
    "timm-res2net101_26w_4s",
    "timm-res2net50_26w_6s",
    "timm-res2net50_26w_8s",
    "timm-res2net50_48w_2s",
    "timm-res2net50_14w_8s",
    "timm-res2next50",
    "timm-regnetx_002",
    "timm-regnetx_004",
    "timm-regnetx_006",
    "timm-regnetx_008",
    "timm-regnetx_016",
    "timm-regnetx_032",
    "timm-regnetx_040",
    "timm-regnetx_064",
    "timm-regnetx_080",
    "timm-regnetx_120",
    "timm-regnetx_160",
    "timm-regnetx_320",
    "timm-regnety_002",
    "timm-regnety_004",
    "timm-regnety_006",
    "timm-regnety_008",
    "timm-regnety_016",
    "timm-regnety_032",
    "timm-regnety_040",
    "timm-regnety_064",
    "timm-regnety_080",
    "timm-regnety_120",
    "timm-regnety_160",
    "timm-regnety_320",
    "timm-gernet_s",
    "timm-gernet_m",
    "timm-gernet_l",
    "senet154",
    "se_resnet50",
    "se_resnet101",
    "se_resnet152",
    "se_resnext50_32x4d",
    "se_resnext101_32x4d",
    "timm-skresnet18",
    "timm-skresnet34",
    "timm-skresnext50_32x4d",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "inceptionresnetv2",
    "inceptionv4",
    "xception",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    "efficientnet-b6",
    "efficientnet-b7",
    "timm-efficientnet-b0",
    "timm-efficientnet-b1",
    "timm-efficientnet-b2",
    "timm-efficientnet-b3",
    "timm-efficientnet-b4",
    "timm-efficientnet-b5",
    "timm-efficientnet-b6",
    "timm-efficientnet-b7",
    "timm-efficientnet-b8",
    "timm-efficientnet-lite0",
    "timm-efficientnet-lite1",
    "timm-efficientnet-lite2",
    "timm-efficientnet-lite3",
    "timm-efficientnet-lite4",
    "mobilenet_v2",
    "timm-mobilenetv3_large_075",
    "timm-mobilenetv3_large_100",
    "timm-mobilenetv3_large_minimal_100",
    "timm-mobilenetv3_small_075",
    "timm-mobilenetv3_small_100",
    "timm-mobilenetv3_small_minimal_100",
    "dpn68",
    "dpn98",
    "dpn131",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]
