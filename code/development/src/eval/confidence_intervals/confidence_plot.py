"""Plot average interval sizes and empirical coverages for varying nominal confidences."""

import json
import subprocess as sp
import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.general_utils import outpath

parser = ArgumentParser()
parser.add_argument(
    "--confidence_intervals_path",
    "-p",
    type=str,
    help="Path of the folder confidence_intervals",
)
parser.add_argument(
    "--plot_avg_interval_size",
    action="store_true",
    help="Whether to generate average interval size plot",
)
parser.add_argument(
    "--plot_empirical_coverage",
    action="store_true",
    help="Whether to generate empirical coverage plot",
)
parser.add_argument(
    "--algorithms",
    nargs="+",
    type=str,
    default=[
        "standard_volume_prediction",
        "standard_volume_prediction_normalized_by_vol",
        "thresholded_volume_prediction",
    ],
    help="Confidence levels.",
)
parser.add_argument(
    "--colors",
    nargs="+",
    type=str,
    help="Colors to be used in plots.",
)
parser.add_argument(
    "--pgf",
    action="store_true",
    help="Whether to generate plots in PGF format instead of default PDF.",
)
parser.add_argument(
    "-c",
    "--confidences",
    nargs="+",
    type=float,
    default=[i / 1000 for i in range(800, 995, 5)],
    help="Confidence levels to use for the plot",
)
args = parser.parse_args()

if not args.plot_avg_interval_size and not args.plot_empirical_coverage:
    print("- Aborting. No plotting argument was given.")
    sys.exit(1)

CONFIDENCES = args.confidences

df_empirical_coverage = pd.DataFrame(index=CONFIDENCES)
df_avg_interval_size = pd.DataFrame()

for algorithm in args.algorithms:
    confidence_intervals_path = f"{args.confidence_intervals_path}/{algorithm}/test_database"
    for confidence in CONFIDENCES:
        with open(
            f"{confidence_intervals_path}/1d_intervals-{algorithm}-c={confidence}.json",
        ) as f:
            file = json.load(f)
        nominal_coverage = file["hparams"]["confidence"]
        empirical_coverages = []
        interval_sizes = []
        for subject in file["data"]:
            empirical_coverages.append(subject["lower_volume"] <= subject["target_volume"] <= subject["upper_volume"])
            interval_sizes.append(abs(subject["upper_volume"] - subject["lower_volume"]) / subject["target_volume"])
        empirical_coverage = np.mean(empirical_coverages)
        avg_interval_size = np.mean(interval_sizes)
        df_empirical_coverage.loc[confidence, algorithm] = empirical_coverage
        df_avg_interval_size.loc[confidence, algorithm] = avg_interval_size

df_avg_interval_size = df_avg_interval_size.sort_index()

# Set output directory

OUTPUT_DIR = outpath("eval/plots/confidence_intervals")

# Plot average interval sizes

NAMES = {
    "standard_volume_prediction": "Algorithm 2 ($g(\\mathcal{X}(X)) = 1$)",
    "standard_volume_prediction_normalized_by_vol": "Algorithm 2 ($g(\\mathcal{X}(X)) = Vol(\\mathcal{M}(X)_{\\geq .5})$)",
    "thresholded_volume_prediction": "Algorithm 1",
    "segmentation_prediction-l=0.05": "Algorithm 3 ($\\lambda = .05$)",
    "segmentation_prediction-l=0.10": "Algorithm 3 ($\\lambda = .1$)",
    "segmentation_prediction-l=0.1": "Algorithm 3 ($\\lambda = .1$)",
}

cm = 1 / 2.54  # centimeters in inches
w = 9
h = 2 * w / 4

plt.rc("font", size=7)
plt.rc("legend", fontsize=6)
plt.rc("axes", labelsize=8)

if args.plot_avg_interval_size:
    df_avg_interval_size = df_avg_interval_size.rename(NAMES, axis=1)

    df_avg_interval_size.plot(color=args.colors, figsize=(w * cm, h * cm), linewidth=0.7)
    plt.xlabel("Nominal confidence")
    plt.ylabel("Avg. interval size\n(norm. by target)")
    if args.pgf:
        plt.savefig(outpath(f"{OUTPUT_DIR}/avg_interval_size.pgf"), bbox_inches="tight")
        sp.call(["sed", "-i", r"s/\\..family//g", outpath(f"{OUTPUT_DIR}/avg_interval_size.pgf")])
    else:
        plt.savefig(outpath(f"{OUTPUT_DIR}/avg_interval_size.pdf"))
    plt.close()

# Prepare coverage data for plotting
if args.plot_empirical_coverage:
    df_empirical_coverage = df_empirical_coverage.rename(NAMES, axis=1)

    # Plot nominal confidence against empirical confidence of all algorithms
    df_empirical_coverage.plot(color=args.colors, figsize=(w * cm, h * cm), legend=False, linewidth=0.7)
    plt.xlabel("Nominal confidence")
    plt.ylabel("Empirical confidence")

    # Draw dashed diagonal line
    plt.axline([1, 1], slope=1, linestyle="dashed", c="black", lw=0.6)

    # Export figure
    if args.pgf:
        plt.savefig(outpath(f"{OUTPUT_DIR}/empirical_confidence.pgf"), bbox_inches="tight")
        sp.call(["sed", "-i", r"s/\\..family//g", outpath(f"{OUTPUT_DIR}/empirical_confidence.pgf")])
    else:
        plt.savefig(outpath(f"{OUTPUT_DIR}/empirical_confidence.pdf"))
    plt.close()
