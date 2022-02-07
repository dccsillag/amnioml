"""
Visualize a slice of a subject in a dataset. Optionally, highlight the target.

Example usage:
    python src/data/visualize_slice.py -d segmentation_876 -s 88f3bf5025667ae0c04dfe52211a5d27 -i 70 -o out.png # just show the exam
    python src/data/visualize_slice.py -d segmentation_876 -s 88f3bf5025667ae0c04dfe52211a5d27 -i 70 -o out.png --target # show the exam and highlight the target segmentation
"""

from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
from imageio import imwrite
from skimage.transform import resize

from src.utils.colors import TARGET
from src.utils.general_utils import blend, grayscale_to_rgba, mask_to_image, normalize_image

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", type=Path, required=True, help="Which dataset to visualize")
parser.add_argument("-s", "--subject_id", required=True, help="Which subject to visualize")
parser.add_argument("-i", "--slice_number", type=int, required=True, help="Which slice to take")
parser.add_argument("-o", "--output", type=str, required=True, help="Where to write the output image to")
parser.add_argument("-t", "--target", action="store_true", help="Highlight the target segmentation")
parser.add_argument("-r", "--resize", type=int, help="Resize the slice to be this size")
args = parser.parse_args()

dataset_path = Path("data") / "processed" / args.dataset

with h5py.File(dataset_path / "full.hdf") as file:
    subject_group = file[args.subject_id]
    slices_group = subject_group["slices"]
    slice_group = slices_group[str(args.slice_number)]

    exam_slice = slice_group["exam"][:]
    assert isinstance(exam_slice, np.ndarray)
    target_slice = slice_group["segmentation"][:]
    assert isinstance(target_slice, np.ndarray)

    exam_slice = grayscale_to_rgba(normalize_image(exam_slice, 0, 1))
    target_slice = mask_to_image(target_slice, TARGET)

    if args.resize is not None:
        exam_slice = resize(exam_slice, (args.resize, args.resize))
        target_slice = resize(target_slice, (args.resize, args.resize))

    components = [exam_slice]
    if args.target:
        components.append(target_slice)

    output_image = 255 * blend(*components)

    imwrite(args.output, output_image)
