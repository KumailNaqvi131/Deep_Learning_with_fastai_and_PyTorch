"""
Chapter 9 — Tabular Modeling Deep Dive
Topic 8: Extrapolation and finding out-of-domain data (book sections "The Extrapolation Problem",
"Finding Out-of-Domain Data")

A random forest predicts averages of training rows: it can never predict a value outside the range
it has seen. For data with a time trend (prices going up), that's a problem for the future.

  1. A toy: y = x + noise. Train a forest on x in 0..15, predict x in 15..20: flat line.
  2. Is our validation set different from the training set? Train a forest to predict "is this row
     in the validation set?". The columns it uses are the ones that differ: the time columns
     (saleElapsed, SalesID, MachineID grow over time). Drop those that don't help, and use only
     recent data (saleYear > 2004).

Run:
    python 08_extrapolation_and_out_of_domain.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from bulldozers_data import get_final_xs, m_rmse, need_sklearn, rf, rf_feat_importance  # noqa: E402
from common import outputs_dir, try_import_matplotlib  # noqa: E402

need_sklearn()
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

OUT = outputs_dir(__file__)

# ---- 1. The extrapolation problem ------------------------------------------------------------------------
torch.manual_seed(42)
x_lin = torch.linspace(0, 20, steps=40)
y_lin = x_lin + torch.randn_like(x_lin)
xs_lin = x_lin.unsqueeze(1)
m_lin = RandomForestRegressor(random_state=42).fit(xs_lin[:30], y_lin[:30])
print("x in the future:      ", x_lin[30:36].round().tolist())
print("true y:               ", y_lin[30:36].round().tolist())
print("random forest predicts:", m_lin.predict(xs_lin[30:36]).round().tolist(), "<- stuck at the last values seen")
plt = try_import_matplotlib()
if plt is not None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.scatter(x_lin, y_lin, s=20, label="data")
    ax.scatter(x_lin[30:], m_lin.predict(xs_lin[30:]), color="red", alpha=0.5, s=20, label="forest, future")
    ax.legend()
    fig.savefig(OUT / "extrapolation.png", bbox_inches="tight")

# ---- 2. Finding out-of-domain data -----------------------------------------------------------------------------
to, xs_final, y, valid_xs_final, valid_y = get_final_xs()
df_dom = pd.concat([xs_final, valid_xs_final])
is_valid = np.array([0] * len(xs_final) + [1] * len(valid_xs_final))
m = rf(df_dom, is_valid)
print("\nColumns that tell training rows from validation rows:\n", rf_feat_importance(m, df_dom)[:6].to_string(index=False))

m = rf(xs_final, y)
print(f"\nvalid RMSE with all columns: {m_rmse(m, valid_xs_final, valid_y):.4f}")
for c in ("SalesID", "saleElapsed", "MachineID"):
    if c in xs_final:
        m = rf(xs_final.drop(c, axis=1), y)
        print(f"   without {c:12s}: {m_rmse(m, valid_xs_final.drop(c, axis=1), valid_y):.4f}")

time_vars = [c for c in ("SalesID", "MachineID") if c in xs_final]
xs_final_time, valid_xs_time = xs_final.drop(time_vars, axis=1), valid_xs_final.drop(time_vars, axis=1)
m = rf(xs_final_time, y)
print(f"without {time_vars}: {m_rmse(m, valid_xs_time, valid_y):.4f}")

sale_year = to.train.xs["saleYear"]
filt = (sale_year > 2004).values
m = rf(xs_final_time[filt], y[filt])
print(f"only sales after 2004 ({filt.sum()} of {len(filt)} rows): {m_rmse(m, valid_xs_time, valid_y):.4f}")
print("Half the data, about the same error: the old sales add little about today's market. (In the book's real\n"
      "data, using only recent sales even improved the model a little.)")
