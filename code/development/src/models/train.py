"""
Train a neural network model.

Example usage:

# trains a small_unet model on dataset segmentation_700 for 50 epochs, using GPU and saving checkpoints
python src/models/train.py small_unet -d segmentation_700 -e 50 --gpu 1 -sc

"""

import argparse
import importlib
import os
import socket
from datetime import datetime as dt

import pytorch_lightning as pl
import pytz as tz

from src.utils.general_utils import make_sync_request
from src.utils.model_utils import (
    DataloaderVolumeRegression,
    HdfDataloader2d,
    TensorBoardLogger,
    get_model_args,
    smp_models,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help="Which model to train",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default="segmentation_20",
        help="Name of dataset to use",
    )
    parser.add_argument(
        "--image_size",
        "-is",
        default=256,
        type=int,
        help="Length and width of resized input",
    )
    parser.add_argument(
        "--num_workers",
        "-nw",
        default=1,
        type=int,
        help="Number of worker processes for background data loading",
    )
    parser.add_argument(
        "--optimizer",
        "-op",
        default="adam",
        choices=["adam", "rmsprop"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--learning_rate",
        "-lr",
        default=1e-3,
        type=float,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        default=4,
        type=int,
        help="Number of batch sizes for data loader",
    )
    parser.add_argument(
        "--num_channels",
        "-nc",
        default=3,
        type=int,
        help="Number of channels to use in each slice",
    )
    parser.add_argument(
        "--channel_stride",
        "-cs",
        default=1,
        type=int,
        help="Stride between channels.",
    )
    parser.add_argument(
        "--loss",
        "-l",
        default="bce",
        type=str,
        help="Metric for training",
    )
    parser.add_argument(
        "--max_epochs",
        "-e",
        default=100,
        type=int,
        help="Maximum number of metrics to use in training",
    )
    parser.add_argument(
        "--save_checkpoints",
        "-sc",
        action="store_true",
        help="Save model checkpoints during training",
    )
    parser.add_argument(
        "--output_to_base_folder",
        "-bf",
        action="store_true",
        help="Save output to base folder instead of personal folder",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        type=int,
        help="Number of GPUs to train on.",
    )
    parser.add_argument(
        "--tpu",
        default=None,
        type=int,
        choices=[None, 1, 8],
        help="How many TPU cores to train on (1 or 8)",
    )
    parser.add_argument(
        "--sync_request",
        "-sr",
        action="store_true",
        help="Make sync request after training",
    )
    parser.add_argument(
        "--normalize_using_center",
        "-nuc",
        default=False,
        type=bool,
        help="Normalize the image using the pixels in the center only, without the borders.",
    )
    parser.add_argument(
        "--tag",
        default="untagged",
        type=str,
        help="Tag to identify the run",
    )
    args, extra_args_list = parser.parse_known_args()

    base_folder = "" if args.output_to_base_folder else "personal"
    pl.seed_everything(0)

    if args.model.startswith("volume_regression."):
        dataloader = DataloaderVolumeRegression(
            dataset=args.dataset,
            image_size=args.image_size,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        )
        num_channels = 1
    else:
        dataloader = HdfDataloader2d(
            dataset=args.dataset,
            image_size=args.image_size,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            n_slices=args.num_channels,
            stride=args.channel_stride,
            normalize_using_center=args.normalize_using_center,
        )
        num_channels = args.num_channels

    if args.model.startswith("smp"):
        model_type, model_name = args.model.split("_")
        assert model_name in smp_models, f"{model_name} not in {smp_models}"
        model_module = importlib.import_module(f"src.models.{model_type}.{model_name.lower()}.model")
    else:
        model_module = importlib.import_module(f"src.models.{args.model}.model")

    if hasattr(model_module, "add_predefined_args"):
        extra_args = vars(get_model_args(extra_args_list, model_module.add_predefined_args))  # type: ignore
    else:
        extra_args = {}
    args = argparse.Namespace(**vars(args), **extra_args)

    model = model_module.Model(  # type: ignore
        loss=args.loss,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        n_channels=num_channels,
        **extra_args,
        args=vars(args),
    )

    logger = pl.loggers.TensorBoardLogger(
        os.path.join(base_folder, "models"),
        name=args.model,
        version=socket.gethostname()
        + "_"
        + args.model
        + "_"
        + dt.now(tz=tz.timezone("America/Sao_Paulo")).isoformat("T", "milliseconds"),
        default_hp_metric=False,
    )

    callbacks = [
        pl.callbacks.EarlyStopping(monitor="val_loss", patience=7, verbose=False, mode="min"),
        TensorBoardLogger(regression=(True if args.loss == "mse" else False)),
    ]

    if args.save_checkpoints:
        checkpoint_filename = "{epoch}-{step}-{val_loss:.2f}-" + args.loss
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename=checkpoint_filename,
            monitor="val_loss",
            save_last=True,
            save_top_k=1,
        )
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        gpus=args.gpu,
        tpu_cores=args.tpu,
        logger=logger,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        checkpoint_callback=args.save_checkpoints,
    )
    trainer.fit(model, dataloader)
    if args.sync_request:
        make_sync_request(os.path.join(base_folder, "models", args.model, logger.version))
