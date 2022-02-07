import argparse
import os
import sys
from secrets import token_hex

import h5py
import nrrd
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.eval_utils import IterIdExamTargetPred

parser = argparse.ArgumentParser(description="save png image to stdout")


parser.add_argument(
    "--prediction_dataset_path",
    "-p",
    default="",
    help="path to hdf with predictions",
)

parser.add_argument(
    "--hide_exam",
    "-he",
    action="store_true",
    help="don't render the exam (also disable transparency)",
)

parser.add_argument(
    "--hide_target",
    "-ht",
    action="store_true",
    help="don't render the target",
)

parser.add_argument(
    "--hide_prediction",
    "-hp",
    action="store_true",
    help="don't render the prediction",
)


args = parser.parse_args()
f = h5py.File(args.prediction_dataset_path, "r")

prediction_folder = os.path.dirname(args.prediction_dataset_path)
extensionless_filename = os.path.basename(os.path.splitext(args.prediction_dataset_path)[0])
img_folder = os.path.join(prediction_folder, extensionless_filename + "-sagittal_plane_images")

if args.hide_exam:
    img_folder += "-hide_exam"

if args.hide_target:
    img_folder += "-hide_target"

if args.hide_prediction:
    img_folder += "-hide_prediction"

os.makedirs(img_folder)

for subject, exam, segmentation, prediction in tqdm(
    IterIdExamTargetPred(args.prediction_dataset_path), desc="saving sagittal plane images"
):

    prediction = np.array((prediction > 0.5), dtype=int)

    middle_layer = exam.shape[0] // 2

    np_image = np.array(
        [exam[middle_layer, :, :], exam[middle_layer, :, :], exam[middle_layer, :, :]], dtype=np.float32
    )

    np_image = (not args.hide_exam) * np_image / np.max(np_image)
    np_image[2, :, :] += (
        (not args.hide_target)
        * (np.array(segmentation[middle_layer, :, :], dtype=np.float32) > 0.5)
        * (0.4 + 0.6 * args.hide_exam)
    )
    np_image[0, :, :] += (
        (not args.hide_prediction)
        * (np.array(prediction[middle_layer, :, :], dtype=np.float32) > 0.5)
        * (0.4 + 0.6 * args.hide_exam)
    )

    np_image = np.moveaxis(np_image, 0, 2)
    np_image = np.minimum(np_image, np.ones(np_image.shape))
    np_image *= 255

    np_image = np.array(np_image, dtype=np.uint8)
    im = Image.fromarray(np_image, mode="RGB")
    im.convert("RGB")
    im.save(f"{img_folder}/{subject}" + ".png", format="png")
