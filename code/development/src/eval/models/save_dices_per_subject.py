import json
import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from src.utils.eval_utils import IterIdExamTargetPred, dice_coefficient
from src.utils.model_utils import get_model_name

parser = ArgumentParser()
parser.add_argument("-p", "--prediction_paths", nargs="+", help="Path to the prediction HDFs")
args = parser.parse_args()

for pred_path in tqdm(args.prediction_paths):
    dices = {
        id: dice_coefficient(pred[:], target[:]) for id, _exam, target, pred in tqdm(IterIdExamTargetPred(pred_path))
    }
    prediction_folder = os.path.dirname(pred_path)
    extensionless_filename = os.path.basename(os.path.splitext(pred_path)[0])
    json_path = os.path.join(prediction_folder, extensionless_filename + "-dices.json")
    with open((json_path), "w") as f:
        f.write(json.dumps(dices, indent=4))
