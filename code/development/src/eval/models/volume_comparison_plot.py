"""
Plot the predicted volume against the real volume.

Example usage:

python src/eval/models/volume_comparison_plot.py -t data/segmentation_700/test.hdf -p <PATH_TO_PREDICTION_HDF>
"""

import json
import os
import subprocess as sp
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.utils.eval_utils import volumes_from_hdf5s

parser = ArgumentParser()
parser.add_argument(
    "--path",
    help="Path to JSON file with target volume, prediction volume and lower and upper bounds.",
)
parser.add_argument(
    "--colored-dots",
    "-cd",
    action="store_true",
    help="Whether to use different colors for lacking and excessive predictions.",
)
parser.add_argument(
    "--colored-bars",
    "-cb",
    action="store_true",
    help="Whether to use different colors for predictive intervals that do or do not cover target volume.",
)
parser.add_argument(
    "--bars",
    action="store_true",
    help="Whether to plot volume prediction bounds as horizontal bars.",
)
parser.add_argument(
    "--pgf",
    action="store_true",
    help="Whether to generate plots in PGF format instead of default PDF.",
)
args = parser.parse_args()

with open(args.path) as f:
    volumes = json.load(f)

df = pd.json_normalize(volumes["data"]).drop("subject_id", axis=1)

cm = 1 / 2.54  # centimeters in inches
w = 8.5
h = 3 * w / 4

plt.rc("font", size=7)
plt.rc("axes", labelsize=8)
fig, ax = plt.subplots(figsize=(w * cm, h * cm))
# plt.rc( "figure", figsize=(w*cm, h*cm))


lower_volume = df["lower_volume"].to_numpy() / 1000
upper_volume = df["upper_volume"].to_numpy() / 1000
prediction_volume = df["prediction_volume"].to_numpy() / 1000
target_volume = df["target_volume"].to_numpy() / 1000

ax.plot([0, 1], [0, 1], color="black", linestyle="dashed", linewidth=0.6, transform=plt.gca().transAxes)

if args.colored_dots:
    col_dots = np.where(prediction_volume <= target_volume, "red", "blue").tolist()
else:
    col_dots = "black"

ax.scatter(prediction_volume, target_volume, marker="o", c=col_dots, s=2)

if args.bars:
    if args.colored_bars:
        col_bars = np.where((target_volume >= lower_volume) & (target_volume <= upper_volume), "blue", "red").tolist()
    else:
        col_bars = ["gray"] * len(prediction_volume)
    # Matplotlib's errorbar does not support `c` to be a list of colors, so individually plot each error bar with desired color by iterating over all points
    for i in range(len(prediction_volume)):
        ax.errorbar(
            prediction_volume[i],
            target_volume[i],
            xerr=[[prediction_volume[i] - lower_volume[i]], [upper_volume[i] - prediction_volume[i]]],
            fmt="none",
            c=col_bars[i],
            alpha=0.8,
            capsize=1.5,
            capthick=0.6,
            linewidth=0.6,
            zorder=-1,
        )

plt.xlabel("AmnioML's predicted volume (L)")
plt.ylabel("Target volume (L)")

lim = max(prediction_volume.max(), target_volume.max()) * 1.1
plt.ylim(0, lim)
plt.xlim(0, lim)

fig.tight_layout()

if args.pgf:
    plt.savefig("volume_comparison.pgf", bbox_inches="tight")
    sp.call(["sed", "-i", r"s/\\..family//g", "volume_comparison.pgf"])
else:
    plt.savefig("volume_comparison.pdf")
plt.close()
