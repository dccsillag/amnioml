import argparse
import json
import os
import re
import socket
import subprocess
from glob import glob
from time import sleep
from typing import Dict, List

import yaml

from src.utils.general_utils import get_eval_folder_path

parser = argparse.ArgumentParser()

parser.add_argument(
    "--use_gpu",
    action="store_true",
    help="use gpu to make predictions",
)

parser.add_argument(
    "--last",
    action="store_true",
    help="use last checkpoint instead of the best",
)

parser.add_argument(
    "--echo_confidence_interval_commands",
    action="store_true",
    help="echo confidence interval commands instead of running them",
)

parser.add_argument(
    "-bf",
    "--output_to_base_folder",
    action="store_true",
    help="save output to base folder instead of personal folder",
)

parser.add_argument("-p", "--path", help="path of the run")

args = parser.parse_args()


is_test_database_2_prediction_path: Dict[bool, str] = {}

# Prediction
for test_database in [True, False]:

    # get variables from the run's path
    run_path = args.path
    run_path = run_path.rstrip("/")
    run_name = os.path.basename(run_path)
    model_name = os.path.basename(os.path.dirname(run_path))
    eval_folder = get_eval_folder_path(
        model_name, run_name, use_shared_folder=args.output_to_base_folder, test_database=test_database
    )
    if not os.path.isdir(eval_folder):
        os.makedirs(eval_folder)
    checkpoints = glob(f"{run_path}/*/*.ckpt")
    if len(checkpoints) == 0:
        raise FileNotFoundError("no checkpoints found in {run_path}")
    best_checkpoint = [p for p in checkpoints if "epoch" in p and "tmp_end" not in p][0]
    last_checkpoint = [p for p in checkpoints if "last" in p][0]
    checkpoint_path = best_checkpoint
    if args.last:
        checkpoint_path = last_checkpoint

    # save predictions
    subprocess_call = ["python3", "src/models/predict.py", "-cp", checkpoint_path]
    if args.use_gpu:
        subprocess_call.append("--use_gpu")
    if test_database:
        subprocess_call.append("--test")
    if args.output_to_base_folder:
        subprocess_call.append("-bf")

    print(f"test_database={test_database}, call=" + str(subprocess_call))
    prediction_path = eval_folder + "/" + os.path.splitext(os.path.basename(checkpoint_path))[0] + "_original_res.hdf5"
    if not os.path.isfile(prediction_path):
        subprocess.run(subprocess_call, check=True)
    else:
        print("Warning: prediction file exists, skipping prediction")

    # save hparams as a json
    with open(f"{run_path}/hparams.yaml", "r") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
        with open(f"{eval_folder}/hparams.json", "w") as g:
            g.write(json.dumps(hparams, indent=4))

    is_test_database_2_prediction_path[test_database] = prediction_path


# Uncertainty Quantification Pipeline
for test_database in [False]:
    for i in range(2):
        confidences_list = [str(x * 0.005 + 0.5) for x in range(50 * i, 50 * (i + 1))]
        prediction_path = is_test_database_2_prediction_path[test_database]
        subprocess_call = [
            "python3",
            "src/confidence_intervals/thresholded_volume_prediction.py",
            "-p",
            prediction_path,
            "-c",
        ] + confidences_list

        if args.echo_confidence_interval_commands:
            print(" ".join(s for s in subprocess_call))
        else:
            print(f"test_database={test_database}, call=" + str(subprocess_call))
            subprocess.run(subprocess_call, check=True)

        leninences = ["0", "0.05", "0.1", "0.2"]

        for leninency in leninences:
            prediction_path = is_test_database_2_prediction_path[test_database]
            subprocess_call = (
                ["python3", "src/confidence_intervals/segmentation_prediction.py", "-p", prediction_path, "-c"]
                + confidences_list
                + ["-l", leninency]
            )

            if args.echo_confidence_interval_commands:
                print(" ".join(subprocess_call))
            else:
                print(f"test_database={test_database}, call=" + str(subprocess_call))
                subprocess.run(subprocess_call, check=True)

        prediction_path = is_test_database_2_prediction_path[test_database]
        subprocess_call = [
            "python3",
            "src/confidence_intervals/standard_volume_prediction.py",
            "-p",
            prediction_path,
            "-c",
        ] + confidences_list

        if args.echo_confidence_interval_commands:
            print(" ".join(s for s in subprocess_call))
        else:
            print(f"test_database={test_database}, call=" + str(subprocess_call))
            subprocess.run(subprocess_call, check=True)

        prediction_path = is_test_database_2_prediction_path[test_database]
        subprocess_call = [
            "python3",
            "src/confidence_intervals/standard_volume_prediction.py",
            "-p",
            prediction_path,
            "--normalization",
            "volume",
            "-c",
        ] + confidences_list

        if args.echo_confidence_interval_commands:
            print(" ".join(s for s in subprocess_call))
        else:
            print(f"test_database={test_database}, call=" + str(subprocess_call))
            subprocess.run(subprocess_call, check=True)


# Images and Videos
prediction_paths = [is_test_database_2_prediction_path[True]]

# confidence_folder = os.path.dirname(is_test_database_2_prediction_path[False]) + "/confidence_intervals/segmentation_prediction-l=0.1/test_database/"
# prediction_paths += [confidence_folder+"upper_bound-c=0.9.hdf5",
#                      confidence_folder+"lower_bound-c=0.9.hdf5"]

for prediction_path in prediction_paths:
    for hiding_list in [[], ["-he"], ["-hp"], ["-ht"], ["-hp", "-ht"], ["-he", "-ht"], ["-he", "-hp"]]:
        # gen images
        subprocess_call: List[str] = ["python3", "src/eval/models/make_images.py", "-p", prediction_path] + hiding_list  # type: ignore
        print(f"test_database={test_database}, call=" + str(subprocess_call))
        subprocess.run(subprocess_call, check=True)

        # gen video
        subprocess_call: List[str] = ["python3", "src/eval/models/make_video.py", "-p", prediction_path] + hiding_list  # type: ignore
        print(f"test_database={test_database}, call=" + str(subprocess_call))
        subprocess.run(subprocess_call, check=True)
