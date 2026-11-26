"""
Chapter 8 — Collaborative Filtering Deep Dive
Topic 4: Weight decay (book section "Weight Decay")

Weight decay (L2 regularization) adds the sum of all the squared weights to the loss:
    loss_with_wd = loss + wd * (parameters**2).sum()
so the gradient gets an extra  2 * wd * parameters, pulling every weight towards 0. Large weights
make sharp, wiggly functions (think of a*x**2 with a big a); keeping weights small limits how
complex the function can get, so the model generalises better.

    learn.fit_one_cycle(5, 5e-3, wd=0.1)

Run:
    python 04_weight_decay.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from common import Learner, MSELossFlat, outputs_dir, set_seed, try_import_matplotlib  # noqa: E402
from common.collab import EmbeddingDotBias  # noqa: E402
from movielens_data import get_dls  # noqa: E402


def dot_product_bias(n_users, n_movies, n_factors=50):
    """The DotProductBias model of 03_collab_from_scratch.py (fastai calls it EmbeddingDotBias)."""
    return EmbeddingDotBias(n_factors, n_users, n_movies, y_range=(0, 5.5))


plt = try_import_matplotlib()
if plt is not None:
    x = np.linspace(-2, 2, 100)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for a in [1, 2, 5, 10, 50]:
        ax.plot(x, a * x ** 2, label=f"a={a}")
    ax.set_ylim(0, 10)
    ax.legend()
    ax.set_title("larger parameters -> narrower, sharper functions")
    fig.savefig(outputs_dir(__file__) / "parabolas.png", bbox_inches="tight")
    print(f"saved {outputs_dir(__file__) / 'parabolas.png'}")

dls = get_dls()
n_users, n_movies = len(dls.classes["user"]), len(dls.classes["title"])
for wd in (0.0, 0.1):
    print(f"\nwd = {wd}")
    set_seed(42)
    learn = Learner(dls, dot_product_bias(n_users, n_movies), loss_func=MSELossFlat())
    learn.fit_one_cycle(5, 5e-3, wd=wd)
print("\nWith wd=0.1 the training loss is higher but the validation loss is lower: less overfitting.")
