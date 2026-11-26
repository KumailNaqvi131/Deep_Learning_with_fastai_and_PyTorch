"""
Chapter 9 — Tabular Modeling Deep Dive
Topic 9: Using a neural network (book section of the same name)

The same data with a neural net: embeddings for the categorical columns, normalised continuous
columns (the Normalize proc; trees didn't need it), and a few linear layers.

    cont_nn,cat_nn = cont_cat_split(df_nn_final, max_card=9000, dep_var=dep_var)
    to_nn = TabularPandas(df_nn_final, procs_nn, cat_nn, cont_nn, splits=splits, y_names=dep_var)
    dls = to_nn.dataloaders(1024)
    learn = tabular_learner(dls, y_range=(8,12), layers=[500,250], n_out=1, loss_func=F.mse_loss)
    learn.fit_one_cycle(5, 1e-2)

saleElapsed must be continuous: a neural net CAN extrapolate a time trend (a tree can't), but only
if the date is a number, not a category.

Run:
    python 09_tabular_neural_net.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn.functional as F  # noqa: E402

from bulldozers_data import dep_var, get_final_xs, get_splits, load_df, r_mse  # noqa: E402
from common import outputs_dir, set_seed  # noqa: E402
from common.tabular import Categorify, FillMissing, Normalize, TabularPandas, cont_cat_split, tabular_learner  # noqa: E402

set_seed(42)
OUT = outputs_dir(__file__)
_, xs_final, _, _, _ = get_final_xs()
cols = [c for c in xs_final.columns if c not in ("SalesID", "MachineID")]     # as in 08_*.py

df_nn = load_df()
df_nn_final = df_nn[cols + [dep_var]]
cont_nn, cat_nn = cont_cat_split(df_nn_final, max_card=9000, dep_var=dep_var)
if "saleElapsed" in cat_nn:
    cat_nn.remove("saleElapsed")
    cont_nn.append("saleElapsed")
print("continuous:", cont_nn)
print("categorical (number of levels):", df_nn_final[cat_nn].nunique().to_dict())

procs_nn = [Categorify, FillMissing, Normalize]
to_nn = TabularPandas(df_nn_final, procs_nn, cat_nn, cont_nn, splits=get_splits(df_nn), y_names=dep_var)
dls = to_nn.dataloaders(1024)
y = to_nn.train.y
print(f"log price range in the training set: {y.min():.2f} .. {y.max():.2f}  -> y_range=(8,12)")

learn = tabular_learner(dls, y_range=(8, 12), layers=[500, 250], n_out=1, loss_func=F.mse_loss, path=OUT)
learn.lr_find(fname=OUT / "tabular_lr_find.png")
learn.fit_one_cycle(5, 1e-2)
preds, targs = learn.get_preds()
print(f"\nneural net valid RMSE: {r_mse(preds.squeeze(), targs.squeeze()):.4f}")
learn.save("nn")
