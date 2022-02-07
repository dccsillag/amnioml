"""
Plots a histogram of the Hausdorff distance between our predictions and the ground truth.

Example usage:

python src/eval/models/hausdorff_distance_histogram.py -p personal/eval/small_unet/congo_small_unet_2021-11-18T08:28:08.179-03:00/epoch=4-step=2404-val_loss=0.05-bce_original_res.hdf5
"""

import os
from argparse import ArgumentParser

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from medpy.io import load
from tqdm import tqdm

from src.utils.data_utils import get_pool_paths_from_subject_id
from src.utils.eval_utils import hausdorff_distance, remove_small_islands

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--prediction_path", help="Path to the predictions HDF of a model")
    parser.add_argument(
        "-s", "--small_island_size", type=int, help="If specified, remove islands smaller than this size"
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

            # print(segmentation.shape)

            # Remove small islands
            if args.small_island_size is not None:
                segmentation = remove_small_islands(segmentation, args.small_island_size)
                prediction = remove_small_islands(prediction, args.small_island_size)

            # Append this optimal threshold
            hausdorff_dists.append(hausdorff_distance(segmentation, prediction))

    sns.histplot(x=hausdorff_dists, kde=True)
    plt.savefig(os.path.join(os.path.dirname(args.prediction_path), "viz", "hausdorff_distance_histogram.png"))
