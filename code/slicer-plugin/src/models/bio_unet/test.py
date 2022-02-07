import argparse
import glob
import os
from typing import List

import pytorch_lightning as pl
import torch
from logrun.utils.general import inpath, outpath

from src.models.bio_unet.dataloader import DataLoader
from src.models.bio_unet.model import UNet

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    "-d",
    default="segmentation_10",
    help="Name of dataset to use",
)
parser.add_argument(
    "--num_workers",
    "-nw",
    default=8,
    type=int,
    help="Number of worker processes for background data loading",
)
parser.add_argument(
    "--batch_size",
    "-bs",
    default=4,
    type=int,
    help="Number of batch sizes for data loader",
)
parser.add_argument(
    "--input_from_personal_folder",
    "-pf",
    action="store_true",
    help="Read model input from personal folder instead of base folder",
)
parser.add_argument(
    "--use_last_model",
    action="store_true",
    help="Whether to use last or the best model trained",
)
parser.add_argument(
    "--model_folder",
    default=None,
    type=str,
    help="Model folder where checkpoints are stored; if None loads the last version",
)
parser.add_argument(
    "--save_files",
    action="store_true",
    help="Whether to save test predictions to file",
)
parser.add_argument(
    "--no_visualization",
    "-nv",
    action="store_true",
    help="Whether to visualize predictions using napari",
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
    choices=[None, 1, 8],
    help="How many TPU cores to train on (1 or 8)",
)
args = parser.parse_args()


pl.seed_everything(0)

base_folder = "personal" if args.input_from_personal_folder else ""
model_name = os.path.basename(os.path.dirname(os.path.realpath(__file__)))

if not args.model_folder:
    latest_version = max(glob.glob(os.path.join(base_folder, "models", model_name, "version_*")), key=os.path.getctime)
    model_folder = latest_version
else:
    model_folder = args.model_folder

print(f"Loading model: {model_folder}")
if args.use_last_model:
    filename_input = os.path.join(model_folder, "checkpoints", "last.ckpt")
else:
    filename_input = glob.glob(os.path.join(model_folder, "checkpoints", "epoch=*.ckpt"))[0]
print(f"Loading checkpoint: {filename_input}")

dataloader = DataLoader(dataset=args.dataset, num_workers=args.num_workers, batch_size=args.batch_size)
model = UNet.load_from_checkpoint(inpath(filename_input))

logger = pl.loggers.TensorBoardLogger(model_folder, name="", version="test")
trainer = pl.Trainer(gpus=args.gpu, tpu_cores=args.tpu, logger=logger)
trainer.test(model, datamodule=dataloader)

if args.save_files:
    output_folder = os.path.join(model_folder, "test_predictions")
    model.eval()
    exams_l: List[torch.tensor] = []
    segmentations_l: List[torch.tensor] = []
    predictions_masks_l: List[torch.tensor] = []
    loss_values_l: List[float] = []

    model.to("cpu")
    with torch.no_grad():
        for subject_data in dataloader.test_dataloader():
            exams, segmentations = subject_data
            predictions = model(exams.to("cpu"))
            predictions_masks = torch.argmax(predictions, dim=1)

            exams_l.append(exams)
            segmentations_l.append(segmentations)
            predictions_masks_l.append(predictions_masks)

    exams_t = torch.cat(exams_l, dim=0)
    segmentations_t = torch.cat(segmentations_l, dim=0)
    predictions_masks_t = torch.cat(predictions_masks_l, dim=0)

    torch.save(exams_t, outpath(f"{output_folder}/exams.pt"))
    torch.save(segmentations_t, outpath(f"{output_folder}/segmentations.pt"))
    torch.save(predictions_masks_t, outpath(f"{output_folder}/predictions_masks.pt"))
