"""
Generates a table with rows corresponding to models and columns corresponding
to losses; the cells report the values of the losses for each model.

Example usage:

python src/eval/models/make_models_over_losses_table.py -t data/segmentation_700/test.hdf -p <PATH_TO_PREDICTION_HDF> # generate the table
"""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.eval_utils import IterIdExamTargetPred, dice_coefficient
from src.utils.model_utils import get_model_name

parser = ArgumentParser()
parser.add_argument("-p", "--prediction_paths", nargs="+", help="Path to the prediction HDFs")
args = parser.parse_args()

res = []

for preds_path in tqdm(args.prediction_paths):
    model_name = preds_path.split("/")[-3]
    model_loss = preds_path.split("-")[-1].split(".")[0].replace("_original_res", "")
    dices = [
        dice_coefficient(pred[:], target[:]) for _id, _exam, target, pred in tqdm(IterIdExamTargetPred(preds_path))
    ]
    dice_mean = np.mean(dices)
    dice_std = np.std(dices)
    res.append([get_model_name(model_name), model_loss, dice_mean, dice_std])
    del model_name, model_loss, dices, dice_mean, dice_std

df = pd.DataFrame(res, columns=["model", "loss", "dice_mean", "dice_std"])

original = df.copy()

df["dice_mean"] = df["dice_mean"].round(3)
df["dice_std"] = df["dice_std"].round(2)

# Present mean and standard deviation as a single column
df["result"] = (
    df["dice_mean"].astype(str).str.ljust(5, "0").str.cat(df["dice_std"].astype(str).str.ljust(4, "0"), sep=r" \pm ")
)
df["result"] = "$" + df["result"] + "$"

# Get index of model that achieved the best performance
idx = df["dice_mean"].idxmax()

# Summarize mean and standard deviation in a single column
df.loc[[idx], "result"] = (
    df.loc[[idx], "dice_mean"]
    .astype(str)
    .str.ljust(6, "0")
    .str.cat(df.loc[[idx], "dice_std"].astype(str).str.ljust(4, "0"), sep=r"\boldsymbol{\pm}")
)
df.loc[[idx], "result"] = r"$\mathbf{" + df["result"] + "}$"

df = df.drop(["dice_mean", "dice_std"], axis=1)
df = df.rename({"model": "Model", "loss": "Loss"}, axis=1)

# Transform each model in its own column
df = df.pivot_table(index="Model", columns=["Loss"], values="result", aggfunc="first")

# Fill missing results with NA
df = df.fillna("NA")

# Rename losses and models
df = df.rename(
    {
        "ac_bce": "AC+BCE",
        "ace": "ACE",
        "bce": "BCE",
        "dice": "Dice",
    },
    axis=1,
)

# Set models as a standard column
df.columns.name = None
df = df.reset_index()

df = df[["Model", "Dice", "BCE", "AC+BCE"]]

# Export as table
df.to_latex("eval/tables/evaluation_table.tex", index=False, escape=False, column_format="c" * df.shape[1])

print(df)
