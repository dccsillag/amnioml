import os
from argparse import ArgumentParser

import h5py
import hdf5plugin
import numpy as np
from medpy.io import load
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

from src.utils.data_utils import get_pool_paths_from_subject_id
from src.utils.eval_utils import hausdorff_distance
from src.utils.general_utils import outpath

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--prediction_path", help="Path to the predictions HDF of a model")
    parser.add_argument(
        "-a",
        "--alpha",
        type=float,
        default=0.05,
        help="Confidence parameter",
    )
    args = parser.parse_args()

    with h5py.File(args.prediction_path, "r") as preds_file:
        subject_ids = list(preds_file.keys())
        subject_ids = [x for x in subject_ids if x != "13be55f0fa50de46beaaea9ba5603360"]

        hausdorff_dists = []
        for subject_id in tqdm(subject_ids, desc="subjects"):
            # Load prediction and segmentation
            segmentation, _ = load(get_pool_paths_from_subject_id(subject_id)[1])
            segmentation = np.where(segmentation >= 0.5, 1, 0).astype(np.uint8)
            prediction = np.array(preds_file[subject_id])
            prediction = np.where(prediction >= 0.5, 1, 0).astype(np.uint8)

            # Append this optimal threshold
            hausdorff_dists.append(hausdorff_distance(segmentation, prediction))

        # Select threshold to use for everyone
        kernel_size = np.quantile(hausdorff_dists, 1 - args.alpha)
        print(f"Kernel size for everyone is: {kernel_size}")
        kernel_size = round(kernel_size)
        print(f"                    rounded: {kernel_size}")

        # Generate upper sets
        upper_sets = {}
        lower_sets = {}
        for subject_id in tqdm(subject_ids, desc="build confidence sets"):
            prediction = np.array(preds_file[subject_id])
            prediction = np.where(prediction >= 0.5, 1, 0).astype(np.uint8)

            upper_sets[subject_id] = binary_dilation(prediction, iterations=kernel_size).astype(np.uint8)
            lower_sets[subject_id] = binary_erosion(prediction, iterations=kernel_size).astype(np.uint8)

    # Save upper sets
    with h5py.File(
        outpath(os.path.join(os.path.dirname(args.prediction_path), "confidence_interval-hausdorff.hdf5")),
        "w",
    ) as file:
        upper_sets_group = file.create_group("upper_sets")
        lower_sets_group = file.create_group("lower_sets")
        for subject_id, upper_set in tqdm(list(upper_sets.items()), desc="saving upper sets"):
            upper_sets_group.create_dataset(subject_id, data=upper_set, **hdf5plugin.Blosc(), track_times=False)
        for subject_id, lower_set in tqdm(list(lower_sets.items()), desc="saving lower sets"):
            lower_sets_group.create_dataset(subject_id, data=lower_set, **hdf5plugin.Blosc(), track_times=False)
