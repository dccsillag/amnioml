"""
This script reads `data/raw/` and produces a processed dataset in
`data/processed/segmentation_{N}/full.hdf`, where `N` is the number passed in
`--number-of-subjects`.
Additionally, a file `data/processed/segmentation_{N}/raw_mapping.json` is
created, mapping the generated subject_ids to their corresponding raw
directories.

Example usage:

python src/data/create_dataset-segmentation.py -n 700 # creates the dataset 'segmentation_700'
"""

import argparse
import json
import os
import re
from glob import glob
from itertools import takewhile
from tempfile import mkstemp
from typing import Dict, OrderedDict, Tuple

import nrrd
import numpy as np
from tqdm import tqdm

from src.utils.data_utils import (
    fix_segmentation_translation,
    get_dimensional_consistency,
    get_md5_from_filepath,
    get_nonzero_volume,
    get_subject_id_from_md5s,
    get_subject_id_from_raw_path,
    linear_part_of_seg_to_exam_transformation_is_identity,
    save_dataset_2d_hdf,
    seg_to_exam_affine_transformatiion_is_identity,
    voxel_coords_from_seg_to_exam_fall_inside_bounds,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-n",
    type=int,
    required=True,
    help="number of subjects to include",
)
parser.add_argument("-sf", "--output_to_shared_folder", const=True, default=False, action="store_const")
parser.add_argument(
    "-b",
    "--blacklist",
    metavar="FILE",
    type=str,
    default="data/blacklist.txt",
    help="blacklist of subject-ids, any reasonable separation scheme should work fine. Note that, in combination with -j, this option may affect the ratio of subjects in each category (train, val, test)",
)

parser.add_argument(
    "-fhpb",
    "--filter_high_pass_for_brightness",
    metavar="BRIGHTNESS",
    type=float,
    help="applies high pass filter for mean segmentation color after normalization by max. Accept values in (0, 1]. Reference value: .2",
)
parser.add_argument(
    "-flpvar",
    "--filter_low_pass_for_variance",
    metavar="VARIANCE",
    type=float,
    help="applies low pass filter for variance of segmentation color after normalization by max. Accept values in (0,infty). Reference value: 0.015",
)
parser.add_argument(
    "-fhpvol",
    "--filter_high_pass_for_volume",
    metavar="VOLUME",
    type=float,
    help="applies high pass filter for volume (mL) of segmentation (0,infty). Reference value: 300",
)
args = parser.parse_args()

assert args.n >= 1, "Argment n must be a positive"

if args.blacklist != "":
    with open(args.blacklist, "r") as f:
        str_blacklist = f.read()
        str_blacklist = re.sub("[^0-9a-f]+", " ", str_blacklist)
        blacklist = [s for s in str_blacklist.split() if len(s) == 32]
else:
    blacklist = []


# Define filters using args
def high_pass_filter_for_brightness(
    exam_data: np.ndarray,
    exam_header: OrderedDict,
    seg_data: np.ndarray,
    seg_header: OrderedDict,
) -> bool:
    if exam_data.shape != seg_data.shape:
        return False
    mean = np.mean(exam_data, where=seg_data != 0, dtype=float) / np.max(exam_data)
    return mean > args.filter_high_pass_for_brightness


def low_pass_filter_for_variance(
    exam_data: np.ndarray,
    exam_header: OrderedDict,
    seg_data: np.ndarray,
    seg_header: OrderedDict,
) -> bool:
    if exam_data.shape != seg_data.shape:
        return False
    maxval = float(np.max(exam_data))
    variance = np.var(exam_data, where=seg_data != 0, dtype=float) / (maxval * maxval)
    return variance < args.filter_low_pass_for_variance


def high_pass_filter_for_volume(
    exam_data: np.ndarray,
    exam_header: OrderedDict,
    seg_data: np.ndarray,
    seg_header: OrderedDict,
) -> bool:
    return get_nonzero_volume(seg_data, seg_header["space directions"]) > args.filter_high_pass_for_volume


# Filters and fixes system:
TESTS = [
    get_dimensional_consistency,
    linear_part_of_seg_to_exam_transformation_is_identity,
    seg_to_exam_affine_transformatiion_is_identity,
    voxel_coords_from_seg_to_exam_fall_inside_bounds,
]

if args.filter_high_pass_for_brightness is not None:
    TESTS.append(high_pass_filter_for_brightness)
if args.filter_low_pass_for_variance is not None:
    TESTS.append(low_pass_filter_for_variance)
if args.filter_high_pass_for_volume is not None:
    TESTS.append(high_pass_filter_for_volume)


def subject_dir_sorting_key(subject_dir: str) -> Tuple[int, str, int]:
    batch_part = os.path.basename(os.path.dirname(subject_dir))
    subject_number = os.path.basename(subject_dir)

    assert batch_part.startswith("lote")
    batch_part = batch_part[4:]
    batch_number = "".join(takewhile(lambda x: x in "0123456789", batch_part))
    batch_extra = batch_part[len(batch_number) :]

    return int(batch_number), batch_extra, int(subject_number)


subject_dirs = sorted(glob("data/raw/*/*"), key=subject_dir_sorting_key)

# dataset: Dict[str, Tuple[np.ndarray, OrderedDict, np.ndarray, OrderedDict]] = {}
raw_ids: Dict[str, str] = {}
with tqdm(total=min(len(subject_dirs), args.n), desc="processing dataset") as progressbar:
    for raw_subject_path in subject_dirs:
        if len(raw_ids) >= args.n:
            break

        # Get the raw data paths for this subject
        path_to_exam = os.path.join(raw_subject_path, "1.nrrd")
        path_to_segmentation = os.path.join(raw_subject_path, "2.nrrd")

        # Load this subject's NRRDs
        try:
            exam_data, exam_header = nrrd.read(path_to_exam)
            segmentation_data, segmentation_header = nrrd.read(path_to_segmentation)
            segmentation_data = (segmentation_data != 0).astype(segmentation_data.dtype)
        except FileNotFoundError:
            print(f"No data in {raw_subject_path}; skipping")
            continue

        subject_id = get_subject_id_from_raw_path(raw_subject_path)

        # Run tests
        if not all(test(exam_data, exam_header, segmentation_data, segmentation_header) for test in TESTS):
            print(f"Tests failed for subject {subject_id}; skipping it")
            continue

        # Consider the blacklist
        if subject_id in blacklist:
            print(f"Subject {subject_id} is in the blacklist; skipping it")
            continue

        # Add it to the dataset
        raw_ids[subject_id] = raw_subject_path
        progressbar.update()

if len(raw_ids) < args.n:
    args.n = len(raw_ids)
    print(f"Warning: Insufficient number of subjects in the dataset, setting n={args.n}")

# Save the output HDF
output_folder = ""
if args.output_to_shared_folder:
    output_folder = "data"
else:
    output_folder = "personal/data"
output_folder += f"/processed/segmentation_{args.n}"
if os.path.isdir(output_folder):
    raise IOError(f"Dataset already exists: {output_folder}")

os.makedirs(output_folder)
with open(f"{output_folder}/raw_mapping.json", "w") as file:
    json.dump(raw_ids, file, indent=4)
save_dataset_2d_hdf(raw_ids, f"{output_folder}/full.hdf")
