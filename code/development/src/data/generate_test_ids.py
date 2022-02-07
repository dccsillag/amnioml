import json
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd

parser = ArgumentParser()
parser.add_argument(
    "--path_raw_mapping",
    "-p",
    type=str,
    default="personal/data/processed/segmentation_750/raw_mapping.json",
    help="Path to mapping of raw IDs to checksums.",
)
parser.add_argument(
    "--number_of_subjects",
    "-n",
    type=int,
    default=112,
    help="Number of subjects to be included in test set.",
)
args = parser.parse_args()

# Load medical reports
amniotic_fluid = pd.read_csv("eval/heron/amniotic_fluid.csv", usecols=["#id", "IDADE", "Líquido Amniótico"])
diseases = pd.read_csv("eval/heron/diseases.csv", usecols=["#id", "IDADE", "Doença/Patologia"])

df = amniotic_fluid.merge(diseases, on="#id")
del amniotic_fluid, diseases

# Compare information disagreement between files
age = df.loc[df["IDADE_x"].notnull() & df["IDADE_y"].notnull(), ["IDADE_x", "IDADE_y"]]
age_disagreement = age[age["IDADE_x"] != age["IDADE_y"]]

# There are two cases where `IDADE_x` and `IDADE_y` do not match
assert len(age_disagreement) == 2

# In both cases, `IDADE_x` has a nonsensical value (121 gestational weeks)
# "The longest recorded pregnancy was 375 days", so less than 54 weeks.
assert (age_disagreement["IDADE_x"] == 121).all()

# `IDADE_y`, on the other hand, has plausible values
assert (age_disagreement["IDADE_y"] == [25, 34]).all()

# There is no `IDADE_x` available for unavailable `IDADE_y`
assert df[df["IDADE_x"].notnull() & df["IDADE_y"].isnull()].empty

# There are 68 cases of unavailable `IDADE_x` where `IDADE_y` is available
assert len(df[df["IDADE_x"].isnull() & df["IDADE_y"].notnull()]) == 68

# We conclude that `IDADE_x` has incorrect and more missing data, so stick to `IDADE_y`
df = df.drop("IDADE_x", axis=1)

# Rename columns and set `id` as index
df = df.rename(
    {
        "#id": "id",
        "Líquido Amniótico": "amniotic_fluid",
        "IDADE_y": "weeks",
        "Doença/Patologia": "disease",
    },
    axis=1,
)

df = df.set_index("id")

# Drop columns where all values are missing
df = df.dropna(axis=0, how="all")

# Normalize columns
df["amniotic_fluid"] = df["amniotic_fluid"].str.lower().str.strip()
df["disease"] = df["disease"].str.lower().str.strip()

df["amniotic_fluid"] = df["amniotic_fluid"].replace(
    {
        "oligdramnia": "oligodramnia",
        "pólidramnia": "polidramnia",
        "nprmal": "normal",
        "normodramnia": "normal",
    }
)

# Subset data to interesting cases
df = df.loc[
    df["amniotic_fluid"].isin(["oligodramnia", "normal", "polidramnia"]) & df["weeks"].notna() & df["disease"].notna()
]

# Map raw IDs to `subject_id`s (checksums)
with open(args.path_raw_mapping) as f:
    ids = json.load(f)

ids = pd.json_normalize(ids).T.reset_index()
ids.columns = ["subject_id", "path"]
ids["id"] = ids["path"].str.split("/").str[-1].astype(int)
ids = ids.drop("path", axis=1).set_index("id")

df = df.merge(ids, how="left", left_index=True, right_index=True)

# Remove exams unavailable on our dataset
df = df[df["subject_id"].notna()]

# Sort `amniotic_fluid` alphabetically and get last values
# The idea is to include oligohydramnios and polyhydramnios cases, not only normal ones
df = df.sort_values("amniotic_fluid").iloc[-args.number_of_subjects :]

# Export resulting subject IDs as csv file
df["subject_id"].to_csv("data/test_subjects.csv", index=False, header=False)
