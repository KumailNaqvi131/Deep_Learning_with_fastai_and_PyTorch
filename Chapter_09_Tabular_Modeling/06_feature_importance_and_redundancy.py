"""
Chapter 9 — Tabular Modeling Deep Dive
Topic 6: Model interpretation: tree variance, feature importance, removing low-importance
variables, removing redundant features (book sections of the same names)

  * Tree variance for prediction confidence: the trees of a forest disagree more on unusual rows;
    the std of their predictions tells you how confident the forest is about each prediction.
  * Feature importance: how much each column's splits reduce the error, over all trees.
  * Remove low-importance columns: a simpler model, just as good, easier to study.
  * Redundant columns: cluster the columns by rank correlation (Spearman); columns merged early
    measure nearly the same thing. Drop one of each pair if the OOB score doesn't get worse.

Run:
    python 06_feature_importance_and_redundancy.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from bulldozers_data import REDUNDANT, get_xs, m_rmse, need_sklearn, rf, rf_feat_importance  # noqa: E402
from common import outputs_dir, try_import_matplotlib  # noqa: E402

need_sklearn()
OUT = outputs_dir(__file__)
to, xs, y, valid_xs, valid_y = get_xs()
m = rf(xs, y)

# ---- Tree variance ---------------------------------------------------------------------------------------
preds = np.stack([t.predict(valid_xs.values) for t in m.estimators_])
preds_std = preds.std(0)
print("std of the trees' predictions for the first 5 validation rows:", preds_std[:5].round(3))
print(f"least confident row: std {preds_std.max():.3f}, most confident: {preds_std.min():.3f}")

# ---- Feature importance ---------------------------------------------------------------------------------
fi = rf_feat_importance(m, xs)
print("\nThe 10 most important columns:\n", fi[:10].to_string(index=False))
plt = try_import_matplotlib()
if plt is not None:
    ax = fi[:30].plot("cols", "imp", "barh", figsize=(10, 8), legend=False)
    ax.invert_yaxis()
    ax.figure.savefig(OUT / "feature_importance.png", bbox_inches="tight")
    print(f"saved {OUT / 'feature_importance.png'}")

# ---- Removing low-importance variables --------------------------------------------------------------------
to_keep = list(fi[fi.imp > 0.005].cols)
print(f"\nkeeping the {len(to_keep)} columns with importance > 0.005")
xs_imp, valid_xs_imp = xs[to_keep], valid_xs[to_keep]
m = rf(xs_imp, y)
print(f"with {len(xs.columns)} columns: valid RMSE {m_rmse(rf(xs, y), valid_xs, valid_y):.4f};  "
      f"with {len(to_keep)}: {m_rmse(m, valid_xs_imp, valid_y):.4f}")

# ---- Removing redundant features ---------------------------------------------------------------------------
have_scipy = False
try:
    import scipy.cluster.hierarchy as hc
    from scipy.spatial.distance import squareform
    have_scipy = True
except ImportError:
    print("[info] scipy not installed: showing correlated pairs instead of the dendrogram (pip install scipy)")

corr = xs_imp.rank().corr().fillna(0).values                  # Spearman rank correlation
if have_scipy and plt is not None:
    z = hc.linkage(squareform(1 - np.abs(corr), checks=False), method="average")
    fig = plt.figure(figsize=(10, 6))
    hc.dendrogram(z, labels=list(xs_imp.columns), orientation="left", leaf_font_size=10)
    fig.savefig(OUT / "cluster_columns.png", bbox_inches="tight")
    print(f"saved {OUT / 'cluster_columns.png'} (columns that merge on the right are nearly the same)")
pairs = [(abs(corr[i, j]), a, b) for i, a in enumerate(xs_imp.columns) for j, b in enumerate(xs_imp.columns) if i < j]
print("most correlated pairs of columns:")
for c, a, b in sorted(pairs, reverse=True)[:6]:
    print(f"   {a:22s} {b:22s} {c:.3f}")


def get_oob(df):
    m = rf(df, y, n_estimators=40, min_samples_leaf=15, max_samples=50000, max_features=0.5)
    return m.oob_score_          # R^2 on the out-of-bag rows (1.0 = perfect)


print(f"\nOOB score with all {len(to_keep)} columns: {get_oob(xs_imp):.4f}")
candidates = [c for c in ("saleYear", "saleElapsed", "ProductGroupDesc", "ProductGroup", "fiModelDesc",
                          "fiBaseModel", "Hydraulics_Flow", "Grouser_Tracks", "Coupler_System") if c in xs_imp]
for c in candidates:
    print(f"   without {c:18s}: {get_oob(xs_imp.drop(c, axis=1)):.4f}")
to_drop = [c for c in REDUNDANT if c in xs_imp]
print(f"without all of {to_drop}: {get_oob(xs_imp.drop(to_drop, axis=1)):.4f}  -> about the same, with fewer columns")

xs_final, valid_xs_final = xs_imp.drop(to_drop, axis=1), valid_xs_imp.drop(to_drop, axis=1)
m = rf(xs_final, y)
print(f"final model, {len(xs_final.columns)} columns: train RMSE {m_rmse(m, xs_final, y):.4f}, "
      f"valid RMSE {m_rmse(m, valid_xs_final, valid_y):.4f}")
(OUT / "final_columns.txt").write_text("\n".join(xs_final.columns))
print("final columns saved to outputs/final_columns.txt (used by the next scripts)")
