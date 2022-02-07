"""
Plot a histogram of by how much we err on volume on a given model, and print where we erred by the most.

Example usage:

python src/eval/models/volume_error_histogram.py -t data/segmentation_700/test.hdf -p <PATH_TO_PREDICTION_HDF> # histogram of y-\\hat{y}
python src/eval/models/volume_error_histogram.py -t data/segmentation_700/test.hdf -p <PATH_TO_PREDICTION_HDF> -n # histogram of (y-\\hat{y})/y
"""

import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from src.utils.eval_utils import IterIdExamTargetPred, volume_from_mask
from src.utils.general_utils import outpath

parser = ArgumentParser()
parser.add_argument("-t", "--target_path", help="Path to the dataset HDF (e.g., data/confidence_652-2d/test.h5)")
parser.add_argument("-p", "--prediction_path", help="Path to the prediction HDF")
parser.add_argument("-n", "--normalize", action="store_true", help="Normalize the values in the histogram")
parser.add_argument("-b", "--n_bins", type=int, default=10, help="Number of biins for the histogram")
args = parser.parse_args()

errors = pd.Series()
for id, _exam, target, pred in tqdm(IterIdExamTargetPred(args.prediction_path, args.target_path)):
    exam_volume = volume_from_mask(target, id)
    pred_volume = volume_from_mask(pred, id)
    error = exam_volume - pred_volume
    if args.normalize:
        error = error / exam_volume
    errors[id] = error

print(errors)

errors.plot.hist(bins=args.n_bins)
normalized_str = "normalized" if args.normalize else "not_normalized"
plt.savefig(
    outpath(
        f"{os.path.dirname(args.prediction_path)}/viz/volume_difference_histogram-{normalized_str}-{args.n_bins}_bins.png"
    )
)
