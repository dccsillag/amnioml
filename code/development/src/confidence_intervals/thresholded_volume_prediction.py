import argparse
import json
import os
from typing import Any, List

import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm

from src.utils.eval_utils import IterIdExamTargetPred, volume

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--prediction_path",
    type=str,
    help="Path to a model's predictions on calibration data.",
)

parser.add_argument(
    "-c",
    "--confidences",
    nargs="+",
    type=float,
    default=[0.9],
    help="Confidence levels.",
)

args = parser.parse_args()

# Set output directory
output_dir = os.path.dirname(args.prediction_path) + "/confidence_intervals/thresholded_volume_prediction"
os.makedirs(output_dir, exist_ok=True)

# Compute thresholds for each subject
per_subject_thresholds = []
print("Computing thresholds for subjects")
for subject_id, _, target, pred in tqdm(IterIdExamTargetPred(args.prediction_path)):
    assert set(np.unique(target)).issubset(set([0, 1])), "There are values in the target that are neither zero nor one."
    proportion_not_segmented = 1 - np.count_nonzero(target) / (target.shape[0] * target.shape[1] * target.shape[2])
    best_threshold = np.quantile(pred, proportion_not_segmented)
    threshold_data = {
        "best_threshold": best_threshold,
        "subject_id": subject_id,
    }
    per_subject_thresholds.append(threshold_data)

# Save thresholds in a json file
with open(output_dir + "/threshold_data.json", "w") as f:
    f.write(json.dumps(per_subject_thresholds, indent=4))

# Compute lower and upper bound thresholds for the given confidence levels
upper_bound_thresholds = []
lower_bound_thresholds = []
for confidence in args.confidences:
    alpha = 1 - confidence
    upper_bound_thresholds.append(np.quantile([x["best_threshold"] for x in per_subject_thresholds], alpha / 2))
    lower_bound_thresholds.append(np.quantile([x["best_threshold"] for x in per_subject_thresholds], 1 - alpha / 2))


def save_intervals(prediction_path: str, output_dir: str, dataset_name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    upper_bound_files = []
    lower_bound_files = []
    for confidence in args.confidences:
        upper_bound_files.append(h5py.File(output_dir + f"/upper_bound-c={confidence}.hdf5", "w"))
        lower_bound_files.append(h5py.File(output_dir + f"/lower_bound-c={confidence}.hdf5", "w"))

    print()
    print(f"Generating lower and upper bounds ({dataset_name} dataset)")
    interval_data: List[Any] = []
    for i in range(len(args.confidences)):
        interval_data.append([])

    for subject_id, _, target, pred, transform, _, _, _ in tqdm(IterIdExamTargetPred(prediction_path, metadata=True)):
        for j in range(len(args.confidences)):
            upper_bound_mask = np.array(pred > upper_bound_thresholds[j], dtype=np.uint8)
            lower_bound_mask = np.array(pred > lower_bound_thresholds[j], dtype=np.uint8)

            interval_entry = {}
            interval_entry["upper_volume"] = volume(upper_bound_mask, transform)
            interval_entry["lower_volume"] = volume(lower_bound_mask, transform)
            interval_entry["prediction_volume"] = volume((pred > 0.5), transform)
            interval_entry["target_volume"] = volume(target, transform)
            interval_entry["subject_id"] = subject_id
            interval_data[j].append(interval_entry)

            upper_bound_files[j].create_dataset(
                subject_id, data=upper_bound_mask, **hdf5plugin.Blosc(), track_times=False
            )
            lower_bound_files[j].create_dataset(
                subject_id, data=lower_bound_mask, **hdf5plugin.Blosc(), track_times=False
            )

    for i, confidence in enumerate(args.confidences):
        json_dictionary = {}
        json_dictionary["hparams"] = {"confidence": confidence}
        json_dictionary["data"] = interval_data[i]
        with open(output_dir + f"/1d_intervals-thresholded_volume_prediction-c={confidence}.json", "w") as f:
            f.write(json.dumps(json_dictionary, indent=4))

    for i in range(len(args.confidences)):
        upper_bound_files[i].close()
        lower_bound_files[i].close()


# validation dataset
save_intervals(args.prediction_path, output_dir, "validation")

# test dataset
test_prediction_path = (
    os.path.dirname(args.prediction_path) + "/test_database/" + os.path.basename(args.prediction_path)
)

save_intervals(test_prediction_path, output_dir + "/test_database", "test")
