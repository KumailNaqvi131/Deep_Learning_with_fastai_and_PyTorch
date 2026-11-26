"""
Chapter 9 — Tabular Modeling Deep Dive
Topic 5: Random forests and out-of-bag error (book sections "Random Forests", "Creating a Random
Forest", "Out-of-Bag Error")

Bagging (Breiman, 1994): train many models, each on a random subset of the rows, and average their
predictions. Each model's errors are mostly uncorrelated, so they cancel out. A random forest =
bagged decision trees, where each split also looks at a random subset of the columns.

    def rf(xs, y, n_estimators=40, max_samples=200_000, max_features=0.5, min_samples_leaf=5):
        return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_samples=max_samples,
                                     max_features=max_features, min_samples_leaf=min_samples_leaf,
                                     oob_score=True).fit(xs, y)

Out-of-bag (OOB) error: each row is predicted only by the trees that did NOT see it in training:
a validation score for free, without a validation set.

Run:
    python 05_random_forests.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from bulldozers_data import get_xs, m_rmse, need_sklearn, r_mse, rf  # noqa: E402
from common import outputs_dir, try_import_matplotlib  # noqa: E402

need_sklearn()
to, xs, y, valid_xs, valid_y = get_xs()

m = rf(xs, y)
print(f"Random forest (40 trees): train RMSE {m_rmse(m, xs, y):.4f}   valid RMSE {m_rmse(m, valid_xs, valid_y):.4f}")

preds = np.stack([t.predict(valid_xs.values) for t in m.estimators_])
print(f"average of the trees' predictions = the forest's prediction: {r_mse(preds.mean(0), valid_y):.4f}")
errs = [r_mse(preds[:i + 1].mean(0), valid_y) for i in range(40)]
print("valid RMSE using the first 1, 5, 10, 20, 40 trees:", [errs[i] for i in (0, 4, 9, 19, 39)],
      "-> more trees help, with diminishing returns")

print(f"\nOOB RMSE: {r_mse(m.oob_prediction_, y):.4f} (computed on the training set, no validation set needed)")
print("OOB error is a bit higher than the validation error here... if it were much LOWER, the validation set"
      " would be different from the training data (here: it's in the future).")

plt = try_import_matplotlib()
if plt is not None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(range(1, 41), errs)
    ax.set_xlabel("number of trees")
    ax.set_ylabel("valid RMSE")
    fig.savefig(outputs_dir(__file__) / "rmse_vs_trees.png", bbox_inches="tight")
