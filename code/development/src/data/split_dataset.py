"""
This script builds upon the `data/processed/segmentation_{N}/full.hdf` file produced by
`create_dataset-segmentation.py` by splitting it into three files:
`data/processed/segmentation_{N}/{train,val,test}.hdf`

Example usage:

python src/data/split_dataset.py personal/processed/segmentation_700 # generates train-val-test split for the dataset in 'personal/processed/segmentation_700'
"""

import csv
import json
import os
import random
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.utils.data_utils import load_whole_subject, save_dataset_2d_hdf

parser = ArgumentParser()
parser.add_argument("dataset_dir", help="path to the dataset directory (e.g., data/segmentation_500)")
parser.add_argument(
    "-u",
    "--ungroup",
    action="store_true",
    help="allow ungrouping of subjects from the same gestation",
)
args = parser.parse_args()

raw_ids = {}
with open(os.path.join(args.dataset_dir, "raw_mapping.json"), "r") as f:
    raw_ids = json.load(f)

subject_ids = list(raw_ids.keys())
groups: List[List[str]] = [[k] for k in subject_ids]

# Group subjects by gestation
if not args.ungroup:
    unique_ids = []
    max_id = 0

    max_dataset_raw_id = max([int(os.path.basename(raw_ids[k])) for k in raw_ids.keys()])

    with open("data/csv/gestational_age.csv") as f:
        spamreader = csv.reader(f, delimiter=";")
        for row in list(spamreader)[1:]:
            max_id = max(max_id, int(row[0]))
            if row[3][0] == "N":
                unique_ids.append(int(row[0]))
    raw_groups = []
    for i in range(len(unique_ids) - 1):
        raw_groups.append(list(range(unique_ids[i], unique_ids[i + 1])))
    raw_groups.append(list(range(unique_ids[-1], max_id + 1)))

    # assume subjects outside table are isolated
    for i in range(max_id + 1, max_dataset_raw_id + 1):
        raw_groups.append([i])

    # print(raw_groups)

    raw_id_2_ids: Dict[int, List[str]] = {i: [] for i in range(1, max_id + 1 + 1)}
    for k in raw_ids.keys():
        raw_id = int(os.path.basename(raw_ids[k]))
        id = k
        raw_id_2_ids[raw_id].append(k)

    groups = [sum([raw_id_2_ids[raw_id] for raw_id in g], []) for g in raw_groups]
    groups = [sorted(g) for g in groups if len(g) >= 1]


# Sort groups by the first subject_id
def group_sorting_key(group: List[str]) -> str:
    return group[0]


groups = sorted(groups, key=group_sorting_key)

# Split groups
n_val = round(len(groups) * 0.12)
n_test = n_val
n_train = len(groups) - n_val - n_test

random.seed(0)
random.shuffle(groups)

train_idx_max = n_train
val_idx_max = train_idx_max + n_val

dataset_groups: Dict[str, List[List[str]]] = {}
dataset_groups["train"] = groups[:train_idx_max]
dataset_groups["val"] = groups[train_idx_max:val_idx_max]
dataset_groups["test"] = groups[val_idx_max:]

with open(f"{args.dataset_dir}/groups.json", "w") as file:
    json.dump(dataset_groups, file, indent=4)

dataset_ids: Dict[str, List[str]] = {}
dataset_ids["train"] = sum(dataset_groups["train"], [])
dataset_ids["val"] = sum(dataset_groups["val"], [])
dataset_ids["test"] = sum(dataset_groups["test"], [])

with open(f"{args.dataset_dir}/dataset.json", "w") as file:
    json.dump(dataset_ids, file, indent=4)

print("number of unique gestations:")
print("train = " + str(len(dataset_groups["train"])) + " (" + str(len(dataset_ids["train"])) + " exams)")
print("val = " + str(len(dataset_groups["val"])) + " (" + str(len(dataset_ids["val"])) + " exams)")
print("test = " + str(len(dataset_groups["test"])) + " (" + str(len(dataset_ids["test"])) + " exams)")
print()

# Save datasets
for name, sub_dataset in [
    ("train", dataset_ids["train"]),
    ("val", dataset_ids["val"]),
    ("test", dataset_ids["test"]),
]:
    print(name + "...")
    sub_raw_ids = {}
    for subject in sub_dataset:
        sub_raw_ids[subject] = raw_ids[subject]

    save_dataset_2d_hdf(sub_raw_ids, os.path.join(args.dataset_dir, f"{name}.hdf"))
