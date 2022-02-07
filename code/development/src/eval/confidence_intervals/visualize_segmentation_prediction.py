"""
Visualize a slice of a prediction. Optionally, highlight correct, missing and excess voxels.
"""

from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
from imageio import imwrite
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.transform import resize

from src.utils.colors import CONFIDENCE, CONTOUR, EXCESS, MISSING, PREDICTION
from src.utils.general_utils import blend, grayscale_to_rgba, mask_to_image, normalize_image

parser = ArgumentParser()
parser.add_argument("-p", "--prediction", type=Path, required=True, help="Path to prediction HDF")
parser.add_argument(
    "-cu", "--confidence_regions_upper", type=Path, required=True, help="Path to upper confidece masks HDF"
)
parser.add_argument(
    "-cl", "--confidence_regions_lower", type=Path, required=True, help="Path to upper confidece masks HDF"
)
parser.add_argument("-s", "--subject_id", required=True, help="Which subject to visualize")
parser.add_argument("-i", "--slice_number", type=int, default=192, help="Which slice to take")
parser.add_argument("-o", "--output", type=str, required=True, help="Where to write the output image to")
parser.add_argument("-r", "--rescale", type=int, default=1, help="Rescale the input images by this amount")
parser.add_argument(
    "--transpose", action="store_true", help="Transpose the exam so that we get a slice parallel to the saggital axis"
)
parser.add_argument("-c", "--crop", help="Crop the output image")
args = parser.parse_args()


def rescale(img: np.ndarray, amount: int) -> np.ndarray:
    if img.dtype == bool:
        img = np.where(img, 255, 0)
    return resize(img, tuple([x * amount for x in img.shape[:2]]), preserve_range=True)


with h5py.File(args.prediction) as file_prediction:
    with h5py.File(args.confidence_regions_lower) as file_confidence_regions_lower:
        with h5py.File(args.confidence_regions_upper) as file_confidence_regions_upper:
            if args.subject_id not in file_prediction:
                print("available subjects:")
                for subject_id in file_prediction.keys():
                    print(f" - {subject_id}")
            assert file_prediction.keys() == file_confidence_regions_lower.keys()
            assert file_prediction.keys() == file_confidence_regions_upper.keys()
            subject_group = file_prediction[args.subject_id]

            if args.transpose:
                index_slice = (args.slice_number, slice(None), slice(None))
            else:
                index_slice = (slice(None), args.slice_number, slice(None))

            exam_slice = subject_group["exam"][:][index_slice]
            assert isinstance(exam_slice, np.ndarray)
            target_slice = subject_group["segmentation"][:][index_slice] >= 0.5
            assert isinstance(target_slice, np.ndarray)

            upper_mask = file_confidence_regions_upper[args.subject_id][:][index_slice] >= 0.5
            lower_mask = file_confidence_regions_lower[args.subject_id][:][index_slice] >= 0.5

            exam_slice = rescale(exam_slice, args.rescale)
            target_slice = rescale(target_slice, args.rescale) >= 0.5
            upper_mask = rescale(upper_mask, args.rescale) >= 0.5
            lower_mask = rescale(lower_mask, args.rescale) >= 0.5

            confidence_region = upper_mask & (~lower_mask)
            target_contour = binary_dilation(target_slice) & (~target_slice)
            # & (~binary_erosion(target_slice))

            out = grayscale_to_rgba(normalize_image(exam_slice, 0, 1))[..., :3]
            out[..., 0] += confidence_region * 0.1
            out[..., 1] += confidence_region * 0.4
            out[..., 2] += confidence_region * 0.1
            # out[..., 2] += target_contour * 0.4
            out[..., 0] = np.where(target_contour, 0.3, out[..., 0])
            out[..., 1] = np.where(target_contour, 0.4, out[..., 1])
            out[..., 2] = np.where(target_contour, 1.0, out[..., 2])

            out = 255 * np.clip(out, 0, 1)

            if args.crop:
                print(out.shape)
                out = out[eval(args.crop)]

            imwrite(args.output, out)
