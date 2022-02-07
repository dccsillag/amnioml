"""
Generates a table reporting the number of parameters for each model.

Example usage:

python src/eval/models/make_n_params_table.py
"""

import argparse
import importlib

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_size",
    "-is",
    default=256,
    type=int,
    help="Number of channels to use in each slice",
)
parser.add_argument(
    "--num_channels",
    "-nc",
    default=3,
    type=int,
    help="Number of channels to use in each slice",
)
parser.add_argument(
    "-m",
    "--models",
    nargs="+",
    help="Which models to include",
)
args = parser.parse_args()

df = pd.DataFrame(columns=["model name", "number of parameters"])

for m in tqdm(args.models):
    model_module = importlib.import_module(f"src.models.{m}.model")
    model = model_module.Model(  # type: ignore
        loss="bce",
        optimizer="adam",
        learning_rate=1e-3,
        image_size=args.image_size,
        n_channels=args.num_channels,
        args=vars(args),
    )

    df = df.append(
        {
            "model name": m,
            "number of parameters": sum([param.nelement() for param in model.parameters()]),
        },
        ignore_index=True,
    )

    del model_module, model

df = df.sort_values("number of parameters")

df = df.set_index("model name")
df.to_latex("eval/tables/number_of_parameters.tex", index=False, escape=False, column_format="c" * df.shape[1])

print(df)
