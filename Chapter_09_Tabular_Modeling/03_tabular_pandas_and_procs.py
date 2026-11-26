"""
Chapter 9 — Tabular Modeling Deep Dive
Topic 3: Using TabularPandas and TabularProc (book section of the same name)

Two "procs" prepare the table:
  * Categorify: replace each category by a number (its index in to.classes[col]; 0 means missing)
  * FillMissing: replace missing numbers by the median, and add a <col>_na column saying it was missing
The validation set is the most recent sales (after October 2011), like Kaggle's test set:

    cond = (df.saleYear<2011) | (df.saleMonth<10)
    splits = (list(train_idx), list(valid_idx))
    cont,cat = cont_cat_split(df, 1, dep_var=dep_var)
    to = TabularPandas(df, procs, cat, cont, y_names=dep_var, splits=splits)

Run:
    python 03_tabular_pandas_and_procs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bulldozers_data import get_splits, get_to, load_df  # noqa: E402

df = load_df()
train_idx, valid_idx = get_splits(df)
print(f"training rows: {len(train_idx)} (until Sep 2011), validation rows: {len(valid_idx)} (Oct 2011 on)")

to = get_to(df)
print(f"\n{len(to.cat_names)} categorical columns:", to.cat_names)
print(f"{len(to.cont_names)} continuous columns:", to.cont_names)
print(f"train: {len(to.train)}, valid: {len(to.valid)}")

print("\nThe processed data, shown with the category names:")
to.show(3)
print("\n... and as it really is (numbers only):")
print(to.items[["state", "ProductGroup", "Drive_System", "Enclosure", "SalePrice"]].head(3).to_string())
print("\nto.classes['ProductSize'] =", to.classes["ProductSize"])
print("(the order is the one we gave the ordinal category, with '#na#' for missing first)")
print("A column created by FillMissing: MachineHoursCurrentMeter_na ->",
      to.items["MachineHoursCurrentMeter_na"].value_counts().to_dict())
