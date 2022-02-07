"""Standard Volume Prediction with optional normalization by volume or total variation."""

import json
import os
from argparse import ArgumentParser
from typing import Any, Dict

import numpy as np
from tqdm import tqdm

from src.utils.data_utils import volume_from_mask
from src.utils.eval_utils import IterIdExamTargetPred

parser = ArgumentParser()
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
    default=[0.9, 0.95],
    help="Confidence levels.",
)

parser.add_argument(
    "-n",
    "--normalization",
    type=str,
    choices=["volume"],
    help="Whether to perform volume normalization.",
)

args = parser.parse_args()

# Compute calibration metrics for all subjects
calibration_metrics_for_all_subjects = []
for subject_id, _, target, pred in tqdm(IterIdExamTargetPred(args.prediction_path)):
    assert set(np.unique(target)).issubset(set([0, 1])), "There are values in the target that are neither zero nor one."
    target_volume = volume_from_mask(target, subject_id)
    prediction_volume = volume_from_mask((pred > 0.5), subject_id)
    subject_metrics = {
        "target_volume": target_volume,
        "prediction_volume": prediction_volume,
        "subject_id": subject_id,
    }
    calibration_metrics_for_all_subjects.append(subject_metrics)

# Compute test metrics for all subjects
test_prediction_path = (
    os.path.dirname(args.prediction_path) + "/test_database/" + os.path.basename(args.prediction_path)
)
test_metrics_for_all_subjects = []
for subject_id, _, target, pred in tqdm(IterIdExamTargetPred(test_prediction_path)):
    assert set(np.unique(target)).issubset(set([0, 1])), "There are values in the target that are neither zero nor one."
    target_volume = volume_from_mask(target, subject_id)
    prediction_volume = volume_from_mask((pred > 0.5), subject_id)
    subject_metrics = {
        "target_volume": target_volume,
        "prediction_volume": prediction_volume,
        "subject_id": subject_id,
    }
    test_metrics_for_all_subjects.append(subject_metrics)

# Set suffix to be appended to output directory and file
if not args.normalization:
    normalization_suffix = ""
elif args.normalization == "volume":
    normalization_suffix = "_normalized_by_vol"

# Calculate interval radii (conformal score)
radii = []

for metrics in calibration_metrics_for_all_subjects:
    if not args.normalization:
        normalization_factor = 1
    elif args.normalization == "volume":
        normalization_factor = metrics["prediction_volume"]

    radii.append(np.abs(metrics["prediction_volume"] - metrics["target_volume"]) / normalization_factor)

    del normalization_factor

for confidence in tqdm(args.confidences):
    radius = np.quantile(radii, confidence)

    for dataset in ["val", "test"]:
        output_dir = (
            os.path.dirname(args.prediction_path)
            + f"/confidence_intervals/standard_volume_prediction{normalization_suffix}"
        )

        if dataset == "val":
            metrics_for_all_subjects = calibration_metrics_for_all_subjects

        elif dataset == "test":
            metrics_for_all_subjects = test_metrics_for_all_subjects
            output_dir += "/test_database"

        os.makedirs(output_dir, exist_ok=True)

        # Calculate lower and upper volumes
        data = []
        for metrics in metrics_for_all_subjects:
            if not args.normalization:
                normalization_factor = 1
            elif args.normalization == "volume":
                normalization_factor = metrics["prediction_volume"]

            interval_metrics = {}

            interval_metrics["lower_volume"] = metrics["prediction_volume"] - radius * normalization_factor
            interval_metrics["upper_volume"] = metrics["prediction_volume"] + radius * normalization_factor
            del normalization_factor

            interval_metrics["prediction_volume"] = metrics["prediction_volume"]
            interval_metrics["target_volume"] = metrics["target_volume"]
            interval_metrics["subject_id"] = metrics["subject_id"]
            data.append(interval_metrics)

        # Save results as JSON file
        json_path = output_dir + f"/1d_intervals-standard_volume_prediction{normalization_suffix}-c={confidence}.json"
        json_dictionary: Dict[str, Any] = {}
        json_dictionary["hparams"] = {"confidence": confidence}
        json_dictionary["data"] = data

        with open(json_path, "w") as json_file:
            json.dump(json_dictionary, json_file, indent=4)
