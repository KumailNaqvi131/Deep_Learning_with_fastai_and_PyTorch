"""
Chapter 9 — Tabular Modeling Deep Dive
Topic 2: Beyond deep learning; the dataset; Kaggle; look at the data; handling dates (book sections
"Beyond Deep Learning", "The Dataset", "Kaggle Competitions", "Look at the Data", "Handling Dates")

For tabular data, ensembles of decision trees (random forests, gradient boosting) are often as
good as neural nets, faster, and easier to interpret. Neural nets win when there are high-
cardinality categorical variables or unstructured data (text, images) in the table.

The dataset: Kaggle's Blue Book for Bulldozers: predict the auction sale price of a piece of
heavy equipment from its type, size, age, options, and the sale date. Metric: root mean squared
log error, so we take the log of the price and use RMSE.

Steps here: read the CSV, tell pandas that ProductSize is an *ordinal* category (it has an order),
take the log of the price, and replace the date by many columns with add_datepart.

Run:
    python 02_look_at_the_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bulldozers_data import dep_var, path, sizes  # noqa: E402
from common.tabular import add_datepart  # noqa: E402

print("files:", sorted(p.name for p in path.iterdir() if not p.name.startswith(".")))
df = pd.read_csv(path / "TrainAndValid.csv", low_memory=False)
print(f"\n{len(df)} rows, {len(df.columns)} columns:\n", list(df.columns))
print("\nProductSize values:", df["ProductSize"].unique().tolist())

df["ProductSize"] = df["ProductSize"].astype("category")
df["ProductSize"] = df["ProductSize"].cat.set_categories(sizes, ordered=True)
print("as an ordered category:", list(df["ProductSize"].cat.categories))

print(f"\nSalePrice: min {df[dep_var].min():.0f}, median {df[dep_var].median():.0f}, max {df[dep_var].max():.0f}")
df[dep_var] = np.log(df[dep_var])
print("we predict log(SalePrice) instead; the competition's metric becomes plain RMSE")

df = add_datepart(df, "saledate")
print("\nafter add_datepart('saledate'), the new columns are:",
      " ".join(c for c in df.columns if c.startswith("sale")))
df_test = pd.read_csv(path / "Test.csv", low_memory=False)
df_test = add_datepart(df_test, "saledate")
print(f"\nTest.csv: {len(df_test)} rows, sales from {df_test.saleYear.min()}-{df_test.saleMonth.min():02d} on: "
      "the test set is in the FUTURE of the training data, so our validation set must be too.")
