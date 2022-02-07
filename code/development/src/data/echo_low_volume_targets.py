from src.utils.data_utils import IterIdExamTarget, get_nonzero_volume
from tqdm import tqdm
import argparse
import numpy as np
import json
import os

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--prediction_path", help="Path to the predictions HDF of a model")

args = parser.parse_args()

raw_mapping = {}

with open(os.path.dirname(args.prediction_path)+"/raw_mapping.json", 'r') as f:
    raw_mapping = json.load(f)


for subject, exam, target, exam_transform, exam_origin, target_transform, target_origin in tqdm(IterIdExamTarget(args.prediction_path, metadata=True)):
    volume = get_nonzero_volume(target, target_transform)
    if volume <= 1:
        print(str(volume)+"\t"+subject+"\t"+raw_mapping[subject])
    #volumes.append(get_nonzero_volume(target, target_transform))

#print("average = "+str(np.average(volumes)))
#print("std = "+str(np.std(volumes)))
