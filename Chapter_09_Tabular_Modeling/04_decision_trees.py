"""
Chapter 9 — Tabular Modeling Deep Dive
Topic 4: Decision trees (book sections "Decision Trees", "Creating the Decision Tree",
"Categorical Variables")

A decision tree asks a sequence of yes/no questions (YearMade <= 1991.5? Coupler_System = Yes?)
and predicts the average price of the training rows that end up in the same leaf. Training
greedily picks, at each node, the column and split value that make the two groups most different.

  * with 4 leaves: we can print and read the tree
  * with no limit: perfect on the training set (each leaf ~1 row), bad on validation: overfitting
  * min_samples_leaf=25: each leaf must contain 25+ rows -> much better generalisation

Categorical variables need no one-hot encoding: a tree can isolate a category with splits on its code.

Run:
    python 04_decision_trees.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bulldozers_data import get_to, m_rmse, need_sklearn  # noqa: E402

need_sklearn()
from sklearn.tree import DecisionTreeRegressor, export_text  # noqa: E402

to = get_to()
xs, y = to.train.xs.copy(), to.train.y
valid_xs, valid_y = to.valid.xs.copy(), to.valid.y

m = DecisionTreeRegressor(max_leaf_nodes=4).fit(xs, y)
print("A tree with 4 leaves (value = mean log price of the rows in that node):")
print(export_text(m, feature_names=list(xs.columns), decimals=2))

print("YearMade has a placeholder 1000 for 'unknown':", (xs.YearMade < 1900).sum(), "rows. Replace it by 1950:")
xs.loc[xs["YearMade"] < 1900, "YearMade"] = 1950
valid_xs.loc[valid_xs["YearMade"] < 1900, "YearMade"] = 1950
m = DecisionTreeRegressor(max_leaf_nodes=4).fit(xs, y)
print(export_text(m, feature_names=list(xs.columns), decimals=2))

m = DecisionTreeRegressor(random_state=42).fit(xs, y)
print(f"No limit:            train RMSE {m_rmse(m, xs, y):.4f}   valid RMSE {m_rmse(m, valid_xs, valid_y):.4f}"
      f"   ({m.get_n_leaves()} leaves for {len(xs)} rows: memorised!)")
m = DecisionTreeRegressor(min_samples_leaf=25, random_state=42).fit(xs, y)
print(f"min_samples_leaf=25: train RMSE {m_rmse(m, xs, y):.4f}   valid RMSE {m_rmse(m, valid_xs, valid_y):.4f}"
      f"   ({m.get_n_leaves()} leaves)")
