import argparse
import json
from typing import Any, Optional, OrderedDict

import h5py
import os
import socket
from datetime import datetime as dt
import pytz as tz

import numpy as np
from tqdm import tqdm
from src.utils.model_utils import CommonLightningModule
from src.utils.data_utils import load_whole_subject, get_nrrd_scale
from src.utils.eval_utils import dice_coefficient

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    "-d",
    default="segmentation_20",
    help="name of dataset to use",
)

parser.add_argument(
    "--output_to_base_folder",
    "-bf",
    action="store_true",
    help="save output to base folder instead of personal folder",
)

parser.add_argument(
    "-n",
    "--n_steps",
    default=200,
    help="number of steps for the threshold varying in the interval (0, 1], default is N_STEPS=200",
)

args = parser.parse_args()

base_folder = "" if args.output_to_base_folder else "personal"

output_folder = os.path.join(
    base_folder, "models",
    "thresholding",
        socket.gethostname()+
        "_thresholding_"+
        dt.now(tz=tz.timezone("America/Sao_Paulo")).isoformat("T", "milliseconds")
)

os.makedirs(output_folder)
print("output_folder="+output_folder)

# waiting for fixes
def volume_from_data_and_header(data: np.array, header: OrderedDict) -> float:
    """ Return volume of nonzero elements in mL """
    scale = get_nrrd_scale(header)
    volume = np.count_nonzero(data) * scale[0] * scale[1] * scale[2]
    return volume / 1000.0

eps = 1e-4

train_path = "data/processed/"+args.dataset+"/train.hdf"
test_path = "data/processed/"+args.dataset+"/test.hdf"
val_path = "data/processed/"+args.dataset+"/val.hdf"


for hdf_path in [train_path, test_path, val_path]:
    targets = h5py.File(hdf_path, "r")#["segmentations"]
    precomputed_volumes = []

    # compute and store the volumes in precomputed_volumes
    for subject in tqdm(targets.keys(), desc=os.path.basename(hdf_path).split('.')[0]):
        subject_volumes = {}
        exam_data, exam_header, segmentation_data, segmentation_header = load_whole_subject(targets, subject)

        # normalize by max
        exam_data = exam_data / max(np.max(exam_data), eps)

        prediction_volumes = []
        intersection_volumes = []

        for i in range(args.n_steps):

            # we want to guarantee that (t < exam < max_threshold) == (t < exam), for all t.
            # hence the eps bellow:
            threshold = (i+1) / args.n_steps + eps
            prediction = exam_data > threshold
            prediction_volume = volume_from_data_and_header(prediction, segmentation_header)
            intersection_volume = volume_from_data_and_header(prediction * segmentation_data, segmentation_header)
            prediction_volumes.append(prediction_volume)
            intersection_volumes.append(intersection_volume)

        subject_volumes["subject_id"] = subject
        subject_volumes["target_volume"] = volume_from_data_and_header(segmentation_data, segmentation_header)
        subject_volumes["prediction_volumes"] = prediction_volumes
        subject_volumes["intersection_volumes"] = intersection_volumes
        precomputed_volumes.append(subject_volumes)

    json_path = output_folder + "/"+ os.path.basename(hdf_path).split('.')[0] + "_volumes.json"
    print("output="+json_path)
    print()
    # save volumes in a json file
    with open(json_path, "w") as f:
        f.write(json.dumps(precomputed_volumes, indent=4))

with open(output_folder+"/hparams.json", "w") as f:
    # args isn't JSON serializable, easier to write as a dictionary
    f.write(json.dumps({"dataset":args.dataset,
        "output_to_base_folder":args.output_to_base_folder,
        "n_steps":args.n_steps}, indent=4))

