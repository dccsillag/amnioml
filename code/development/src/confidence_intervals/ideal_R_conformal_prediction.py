import json
import os
from argparse import ArgumentParser
from typing import Any, Dict, List

from tqdm import trange

parser = ArgumentParser()

parser.add_argument(
    "-p",
    "--prediction_metrics_path",
    type=str,
)

parser.add_argument(
    "-c",
    "--confidences",
    nargs="+",
    type=float,
    default=[0.9, 0.95],
)

args = parser.parse_args()

EPS = 0.001

for i in trange(len(args.confidences)):

    for dataset in ["val", "test"]:
        output_dir = os.path.dirname(args.prediction_metrics_path) + "/confidence_intervals/ideal_R_conform_prediction"

        if dataset == "val":
            test_metrics_path = args.prediction_metrics_path

        if dataset == "test":
            test_metrics_path = (
                os.path.dirname(args.prediction_metrics_path)
                + "/test_database/"
                + os.path.basename(args.prediction_metrics_path)
            )
            output_dir += "/test_database"

        os.makedirs(output_dir, exist_ok=True)
        # Open and reading the test JSON file
        test_metrics_file = open(
            test_metrics_path,
        )
        test_metrics_for_all_subjects = json.load(test_metrics_file)

        # calculating the JSON intervals and metrics
        data: List[Any] = []
        for metrics in test_metrics_for_all_subjects:
            radius = abs(metrics["prediction_volume"] - metrics["target_volume"]) + EPS
            interval_metrics = {}
            interval_metrics["lower_volume"] = metrics["prediction_volume"] - radius
            interval_metrics["upper_volume"] = metrics["prediction_volume"] + radius
            interval_metrics["prediction_volume"] = metrics["prediction_volume"]
            interval_metrics["target_volume"] = metrics["target_volume"]
            interval_metrics["subject_id"] = metrics["subject_id"]
            data.append(interval_metrics)

        test_metrics_file.close()

        # Saving JSON
        json_path = output_dir + f"/1d_intervals-ideal_R_conform_prediction-c={args.confidences[i]}.json"
        json_dictionary: Dict[str, Any] = {}
        json_dictionary["hparams"] = {"confidence": args.confidences[i]}
        json_dictionary["data"] = data

        with open(json_path, "w") as json_file:
            json.dump(json_dictionary, json_file, indent=4)
