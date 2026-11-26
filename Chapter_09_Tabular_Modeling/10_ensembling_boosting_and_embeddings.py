"""
Chapter 9 — Tabular Modeling Deep Dive
Topic 10: Ensembling, boosting, and combining embeddings with other methods (book sections of the
same names)

  * Ensembling: a random forest and a neural net make different kinds of errors; average their
    predictions and the result is usually better than either.
  * Boosting: instead of averaging independent trees, fit a small tree, then fit the next tree to
    the *residuals* (what the previous trees got wrong), and so on; the prediction is the sum.
    That's gradient boosting (XGBoost, LightGBM, sklearn's HistGradientBoostingRegressor).
  * Embeddings for other models: the embeddings a neural net learned for each category can be
    used as input features for a random forest (or any model), often improving it.

Run:
    python 10_ensembling_boosting_and_embeddings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from bulldozers_data import dep_var, get_final_xs, get_splits, load_df, m_rmse, need_sklearn, r_mse, rf  # noqa: E402
from common import set_seed  # noqa: E402
from common.tabular import Categorify, FillMissing, Normalize, TabularPandas, cont_cat_split, tabular_learner  # noqa: E402

need_sklearn()
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

set_seed(42)
to, xs_final, y, valid_xs_final, valid_y = get_final_xs()
cols = [c for c in xs_final.columns if c not in ("SalesID", "MachineID")]
xs_time, valid_xs_time = xs_final[cols], valid_xs_final[cols]

# ---- The two models (as in 08 and 09) ----------------------------------------------------------------------
m = rf(xs_time, y)
rf_preds = m.predict(valid_xs_time)
df_nn = load_df()
df_nn_final = df_nn[cols + [dep_var]]
cont_nn, cat_nn = cont_cat_split(df_nn_final, max_card=9000, dep_var=dep_var)
if "saleElapsed" in cat_nn:
    cat_nn.remove("saleElapsed")
    cont_nn.append("saleElapsed")
to_nn = TabularPandas(df_nn_final, [Categorify, FillMissing, Normalize], cat_nn, cont_nn,
                      splits=get_splits(df_nn), y_names=dep_var)
learn = tabular_learner(to_nn.dataloaders(1024), y_range=(8, 12), layers=[500, 250], n_out=1, loss_func=F.mse_loss)
learn.silent = True
learn.fit_one_cycle(5, 1e-2)
nn_preds = learn.get_preds()[0].squeeze().numpy()

# ---- Ensembling ----------------------------------------------------------------------------------------------
print(f"random forest valid RMSE: {r_mse(rf_preds, valid_y):.4f}")
print(f"neural net valid RMSE:    {r_mse(nn_preds, valid_y):.4f}")
ens_preds = (nn_preds + rf_preds) / 2
print(f"ensemble (average):       {r_mse(ens_preds, valid_y):.4f}")
print("(An average helps most when both models are about equally good; here the net is clearly better, so a\n"
      " weighted average, e.g. 0.8*nn + 0.2*rf, would be the next thing to try:"
      f" {r_mse(0.8 * nn_preds + 0.2 * rf_preds, valid_y):.4f})")

# ---- Boosting by hand ----------------------------------------------------------------------------------------
print("\nGradient boosting by hand: each small tree fits the residuals of the trees before it")
pred_tr, pred_va = np.full(len(y), y.mean()), np.full(len(valid_y), y.mean())
for i in range(1, 201):
    t = DecisionTreeRegressor(max_depth=5, random_state=i).fit(xs_time, y - pred_tr)
    pred_tr += 0.1 * t.predict(xs_time)                       # 0.1 = the learning rate ("shrinkage")
    pred_va += 0.1 * t.predict(valid_xs_time)
    if i in (1, 10, 50, 100, 200):
        print(f"   {i:3d} trees: train RMSE {r_mse(pred_tr, y):.4f}  valid RMSE {r_mse(pred_va, valid_y):.4f}")
try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    gb = HistGradientBoostingRegressor(max_iter=300, random_state=42).fit(xs_time, y)
    print(f"sklearn HistGradientBoostingRegressor: valid RMSE {m_rmse(gb, valid_xs_time, valid_y):.4f}")
except ImportError:
    pass

# ---- Combining embeddings with other methods ---------------------------------------------------------------------
print("\nUsing the neural net's embeddings as features for a random forest:")
model = learn.model.cpu()


def with_embeddings(xs, cat_codes):
    out = [xs.reset_index(drop=True)]
    for i, c in enumerate(cat_nn):
        emb = model.embeds[i].weight.detach().numpy()[cat_codes[c].values.astype(int)]
        out.append(pd.DataFrame(emb, columns=[f"{c}_emb{j}" for j in range(emb.shape[1])]))
    return pd.concat(out, axis=1)


xs_emb = with_embeddings(xs_time, to_nn.train.xs[cat_nn])
valid_xs_emb = with_embeddings(valid_xs_time, to_nn.valid.xs[cat_nn])
m_emb = rf(xs_emb, y.values)
print(f"   random forest + embeddings: valid RMSE {m_rmse(m_emb, valid_xs_emb, valid_y):.4f} "
      f"({xs_emb.shape[1]} columns instead of {xs_time.shape[1]})")
