"""
Visualize a slice of a prediction. Optionally, highlight correct, missing and excess voxels.

Example usage:
    python src/eval/models/visualize_prediction_slice.py -p eval/bio_unet/instance-pauloo_bio_unet_2022-01-10T14:06:09.749-03:00/last_original_res.hdf5 -s d0bfd459da04eef177f7ae103575c409 -i 70 -o out.png # just show the prediction
    python src/eval/models/visualize_prediction_slice.py -p eval/bio_unet/instance-pauloo_bio_unet_2022-01-10T14:06:09.749-03:00/last_original_res.hdf5 -s d0bfd459da04eef177f7ae103575c409 -i 70 -o out.png --target # compare the prediction with the target
"""

from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
from imageio import imwrite
from skimage.transform import resize

from src.utils.colors import EXCESS, MISSING, PREDICTION, TARGET
from src.utils.general_utils import blend, grayscale_to_rgba, mask_to_image, normalize_image

parser = ArgumentParser()
parser.add_argument("-p", "--prediction", type=Path, required=True, help="Path to prediction HDF")
parser.add_argument("-s", "--subject_id", required=True, help="Which subject to visualize")
parser.add_argument("-i", "--slice_number", type=int, required=True, help="Which slice to take")
parser.add_argument("-o", "--output", type=str, required=True, help="Where to write the output image to")
parser.add_argument("-t", "--target", action="store_true", help="Highlight the target segmentation")
parser.add_argument("-r", "--resize", type=int, help="Resize the slice to be this size")
args = parser.parse_args()

with h5py.File(args.prediction) as file:
    if args.subject_id not in file:
        print("available subjects:")
        for subject_id in file.keys():
            print(f" - {subject_id}")
    subject_group = file[args.subject_id]

    exam_slice = subject_group["exam"][:][:, :, args.slice_number]
    assert isinstance(exam_slice, np.ndarray)
    target_slice = subject_group["segmentation"][:][:, :, args.slice_number]
    assert isinstance(target_slice, np.ndarray)
    prediction_slice = subject_group["prediction"][:][:, :, args.slice_number] >= 0.5
    assert isinstance(prediction_slice, np.ndarray)

    true_positive = target_slice * prediction_slice
    false_negative = target_slice * (1 - prediction_slice)
    false_positive = (1 - target_slice) * prediction_slice

    exam_slice = grayscale_to_rgba(normalize_image(exam_slice, 0, 1))
    target_slice = mask_to_image(target_slice, TARGET)
    if args.target:
        prediction_slice = (
            mask_to_image(true_positive, PREDICTION)
            + mask_to_image(false_negative, MISSING)
            + mask_to_image(false_positive, EXCESS)
        )
    else:
        prediction_slice = mask_to_image(prediction_slice, PREDICTION)

    if args.resize is not None:
        exam_slice = resize(exam_slice, (args.resize, args.resize))
        target_slice = resize(target_slice, (args.resize, args.resize))
        prediction_slice = resize(prediction_slice, (args.resize, args.resize))

    components = [exam_slice, prediction_slice]

    output_image = 255 * blend(*components)

    imwrite(args.output, output_image)
