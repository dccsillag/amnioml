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
#from src.utils.model_utils import CommonLightningModule
from src.utils.data_utils import load_whole_subject, get_nrrd_scale
from src.utils.eval_utils import dice_coefficient
from src.utils.general_utils import get_eval_folder_path


parser = argparse.ArgumentParser()
parser.add_argument(
    "-bf",
    "--output_to_base_folder",
    action="store_true",
    help="save output to base folder instead of personal folder",
)

parser.add_argument(
    "-hp",
    "--hparams_path",
    help="path to the hparams.json file",
)

parser.add_argument(
    "-d",
    "--dataset_path_overwrite",
    default="",
    help="path to another hdf5 file",
)

parser.add_argument(
    "--test",
    action="store_true",
    help="use test database instead of val",
)

args = parser.parse_args()

run_path = os.path.dirname(args.hparams_path)
run_name = os.path.basename(run_path)

train_args = {}

with open(args.hparams_path,"r") as f:
    train_args = json.loads(f.read())

n_steps = train_args["n_steps"]


dataset_path = f"data/processed/{train_args['dataset']}"
if args.test:
    dataset_path = f"{dataset_path}/test.hdf"
else:
    dataset_path = f"{dataset_path}/val.hdf"

if len(args.dataset_path_overwrite) >= 1:
    dataset_path = args.dataset_path_overwrite

model_name = "thresholding"
output_folder = get_eval_folder_path(model_name, run_name, args.output_to_base_folder, test_database=args.test)
if len(args.dataset_path_overwrite) >= 1:
    output_folder += (
        f"/{os.path.basename(os.path.dirname(args.dataset_path_overwrite))}"
        f"-{os.path.splitext(os.path.basename(args.dataset_path_overwrite))[0]}"
    )

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#print(dataset)
print(train_args)
print("run_name="+run_name)
print("dataset_path="+dataset_path)

train_volumes_path = run_path+"/train_volumes.json"

train_volumes = []

with open(train_volumes_path, "r") as f:
    train_volumes = json.loads(f.read())

best_single_boundary_avg_dice = 0
index_best_single_boudary = -1

best_double_boundary_avg_dice = 0
min_index_best_double_boudary = -1
max_index_best_double_boudary = -1
eps = 1e-4

for min_index in tqdm(range(n_steps - 1), "computing thresholds"):
    single_boundary_avg_dice = 0
    n_subjects = len(train_volumes)
    for subject_data in train_volumes:
        #dice = subject_data["dices"][min_threshold]

        intersection_min = subject_data["intersection_volumes"][min_index]
        prediction_min = subject_data["prediction_volumes"][min_index]
        target = subject_data["target_volume"]

        dice = 2 * (intersection_min) / (prediction_min + target + eps)

        single_boundary_avg_dice += dice / n_subjects

    if single_boundary_avg_dice > best_single_boundary_avg_dice:
        best_single_boundary_avg_dice = single_boundary_avg_dice
        index_best_single_boudary = min_index

    for max_index in range(min_index, n_steps):
        avg_dice = 0
        for subject_data in train_volumes:
            # for min_t < x < max_t
            intersection_max = subject_data["intersection_volumes"][max_index]
            intersection_min = subject_data["intersection_volumes"][min_index]
            prediction_max = subject_data["prediction_volumes"][max_index]
            prediction_min = subject_data["prediction_volumes"][min_index]
            target = subject_data["target_volume"]
            dice = 2 * (intersection_min - intersection_max) / (prediction_min - prediction_max + target + eps)
            avg_dice += dice / n_subjects

        if avg_dice > best_double_boundary_avg_dice:
            min_index_best_double_boudary= min_index
            max_index_best_double_boudary = max_index
            best_double_boundary_avg_dice = avg_dice

print("Single boundary:")
print("  best_avg_dice =", best_single_boundary_avg_dice)
print("  index =",index_best_single_boudary)
single_boundary_thresold = (index_best_single_boudary+1) / n_steps + eps
print("  threshold =",(index_best_single_boudary+1)/single_boundary_thresold)

print()
print("Double boundary:")
print("  best_avg_dice =", best_double_boundary_avg_dice)
print("  min_index =",min_index_best_double_boudary)
print("  max_index =",max_index_best_double_boudary)
double_boundary_min_threshold = (min_index_best_double_boudary+1) / n_steps + eps
print("  min_threshold =", double_boundary_min_threshold)
double_boundary_max_threshold = (min_index_best_double_boudary+1) / n_steps + eps
print("  max_threshold =",double_boundary_max_threshold)

hdf_single_boundary_path = output_folder + "/single_boundary.hdf"
hdf_double_boundary_path = output_folder + "/double_boundary.hdf"

dataset_file = h5py.File(dataset_path, "r")
double_boundary_f = h5py.File(hdf_double_boundary_path, "w")
single_boundary_f = h5py.File(hdf_single_boundary_path, "w")

subject_ids = list(dataset_file.keys())

for subject_id in tqdm(subject_ids, desc="saving"):
    subject_group_single = single_boundary_f.create_group(subject_id)
    subject_group_double = double_boundary_f.create_group(subject_id)

    for subject_group_i in range(2):
        subject_group = [subject_group_single, subject_group_double][subject_group_i]
        subject_group.create_dataset(
            "exam_transform", data=dataset_file[subject_id]["exam_transform"], track_times=False
        )
        subject_group.create_dataset(
            "exam_origin", data=dataset_file[subject_id]["exam_origin"], track_times=False
        )
        subject_group.create_dataset(
            "segmentation_transform", data=dataset_file[subject_id]["segmentation_transform"], track_times=False
        )
        subject_group.create_dataset(
            "segmentation_origin", data=dataset_file[subject_id]["segmentation_origin"], track_times=False
        )

        exam = np.stack(
            [
                dataset_file[subject_id]["slices"][i]["exam"]
                for i in sorted(list(dataset_file[subject_id]["slices"].keys()), key=int)
            ],
            axis=2,
        )
        segmentation = np.stack(
            [
                dataset_file[subject_id]["slices"][i]["segmentation"]
                for i in sorted(list(dataset_file[subject_id]["slices"].keys()), key=int)
            ],
            axis=2,
        )
        subject_group.create_dataset("exam", data=exam, track_times=False)
        subject_group.create_dataset("segmentation", data=segmentation, track_times=False)
        assert exam.shape == segmentation.shape

        pred = None

        # make prediction
        if subject_group_i == 0:
            pred = np.array(exam > single_boundary_thresold, dtype=np.float32)
        else:
            pred = np.array(
                (exam > double_boundary_min_threshold) *
                (exam < double_boundary_max_threshold), dtype=np.float32)


        subject_group.create_dataset("prediction", data=pred, track_times=False)

