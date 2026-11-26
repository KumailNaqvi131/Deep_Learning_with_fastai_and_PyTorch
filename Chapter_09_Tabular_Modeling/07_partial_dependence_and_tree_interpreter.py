"""
Chapter 9 — Tabular Modeling Deep Dive
Topic 7: Partial dependence, data leakage, tree interpreter (book sections of the same names)

  * Partial dependence: "all else being equal, how does the price change with YearMade?" Replace
    YearMade by 1950 in EVERY row, predict, average; then 1951, ...; plot. This removes the effect
    of the other columns that are correlated with YearMade.
  * Data leakage: information in the training data that wouldn't be available at prediction time
    (e.g. a column filled in *after* the outcome). Signs: suspiciously important features, or
    accuracy too good to be true. Feature importance and partial dependence help to find it.
  * Tree interpreter: for ONE row, how much did each column push the prediction up or down
    compared to the average? (bias + sum of contributions = prediction). The book uses the
    treeinterpreter and waterfallcharts libraries; here it's computed directly from the trees.

Run:
    python 07_partial_dependence_and_tree_interpreter.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from bulldozers_data import get_final_xs, need_sklearn, rf  # noqa: E402
from common import outputs_dir, try_import_matplotlib  # noqa: E402

need_sklearn()
OUT = outputs_dir(__file__)
to, xs_final, y, valid_xs_final, valid_y = get_final_xs()
m = rf(xs_final, y)

# ---- Partial dependence ------------------------------------------------------------------------------------
print("ProductSize counts in the validation set:",
      {to.classes["ProductSize"][int(k)]: int(v) for k, v in valid_xs_final["ProductSize"].value_counts().sort_index().items()})
years = np.arange(1950, 2012)
pd_year = []
for yr in years:
    X = valid_xs_final.copy()
    X["YearMade"] = yr
    pd_year.append(m.predict(X).mean())
print("\nPartial dependence of log(price) on YearMade:")
for yr, p in list(zip(years, pd_year))[::10]:
    print(f"   {yr}: {p:.3f}   price ~ ${np.exp(p):,.0f}")
pd_size = {}
for code, name in enumerate(to.classes["ProductSize"]):
    X = valid_xs_final.copy()
    X["ProductSize"] = code
    pd_size[name] = m.predict(X).mean()
print("(the high value at 1950 comes from the 'unknown year' rows we set to 1950: they are ordinary machines)")
print("Partial dependence on ProductSize:", {k: round(float(v), 3) for k, v in pd_size.items()})
print("('#na#' = missing size: in the real data, its price is suspicious -> possible leakage to investigate)")

plt = try_import_matplotlib()
if plt is not None:
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(years, pd_year)
    axs[0].set_xlabel("YearMade")
    axs[0].set_ylabel("average predicted log(price)")
    axs[1].bar(list(pd_size), list(pd_size.values()))
    axs[1].set_ylim(min(pd_size.values()) - 0.2)
    axs[1].tick_params(axis="x", rotation=45)
    fig.savefig(OUT / "partial_dependence.png", bbox_inches="tight")
    print(f"saved {OUT / 'partial_dependence.png'}")


# ---- Tree interpreter --------------------------------------------------------------------------------------
def tree_contributions(tree, X):
    """Walk each row down the tree: every split moves the prediction from the parent node's mean to the
    child node's mean; credit that change to the column used by the split."""
    t = tree.tree_
    values = t.value[:, 0, 0]
    paths = tree.decision_path(X)
    contribs = np.zeros(X.shape)                                       # [rows, columns]
    for i in range(X.shape[0]):
        nodes = paths.indices[paths.indptr[i]:paths.indptr[i + 1]]       # root -> leaf (ids increase)
        for parent, child in zip(nodes[:-1], nodes[1:]):
            contribs[i, t.feature[parent]] += values[child] - values[parent]
    return values[0], contribs


def forest_contributions(m, X):
    res = [tree_contributions(t, X) for t in m.estimators_]
    return np.mean([b for b, _ in res]), np.mean([c for _, c in res], axis=0)


row = valid_xs_final.iloc[:5]
bias, contributions = forest_contributions(m, row)
prediction = m.predict(row)
print(f"\nTree interpreter for the first validation row: bias (average) {bias:.3f}, "
      f"prediction {prediction[0]:.3f} = bias + sum of contributions ({bias + contributions[0].sum():.3f})")
order = np.argsort(-np.abs(contributions[0]))
print("Biggest contributions (a text 'waterfall'):")
for j in order[:8]:
    c = contributions[0, j]
    bar = ("+" if c > 0 else "-") * int(min(40, abs(c) * 100))
    print(f"   {valid_xs_final.columns[j]:22s} {c:+.3f} {bar}")
