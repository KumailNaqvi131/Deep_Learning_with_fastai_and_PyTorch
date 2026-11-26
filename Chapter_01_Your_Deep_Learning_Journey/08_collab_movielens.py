"""
Chapter 1 — Your Deep Learning Journey
Topic 8: Beyond images: recommendation systems (book section "Deep Learning Is Not Just for
Image Classification")

Predict the rating (0.5-5 stars) a user would give a movie, from the ratings everyone gave:
collaborative filtering (explained in Chapter 8). The book's code:

    path = untar_data(URLs.ML_SAMPLE)
    dls = CollabDataLoaders.from_csv(path/'ratings.csv')
    learn = collab_learner(dls, y_range=(0.5,5.5))
    learn.fine_tune(10)
    learn.show_results()

y_range tells the model the range of the target (a little wider than 1-5, see Chapter 8).

Run:
    python 08_collab_movielens.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common import *  # noqa: F401,F403,E402
from common.collab import CollabDataLoaders, collab_learner  # noqa: E402

set_seed(42)
path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path / "ratings.csv", seed=42)
dls.show_batch(5)

learn = collab_learner(dls, y_range=(0.5, 5.5))
learn.fine_tune(10)

# show_results: a few validation ratings next to the model's predictions
xb, yb = dls.valid.one_batch()
with torch.no_grad():
    pred = learn.model(xb)
rows = [(dls.classes["userId"][int(u)], dls.classes["movieId"][int(m)], float(r), float(p))
        for (u, m), r, p in zip(xb[:9].cpu(), yb[:9, 0].cpu(), pred[:9].cpu())]
print(pd.DataFrame(rows, columns=["userId", "movieId", "rating", "rating_pred"]).round(2).to_string())
