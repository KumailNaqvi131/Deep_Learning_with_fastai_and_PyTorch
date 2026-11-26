"""
Chapter 8 — Collaborative Filtering Deep Dive
Topic 7: Bootstrapping a collaborative filtering model; deep learning for collaborative
filtering (book sections of the same names)

Bootstrapping (the cold-start problem): a brand-new user has no ratings, so no learned factors.
Options: give them the average user's factors (or those of a typical user), or ask a few
questions at sign-up and train a model that maps the answers to the embedding.

Deep learning version: instead of a dot product, concatenate the user and movie embeddings and
pass them through a small neural network. Embedding sizes use fastai's rule of thumb
(get_emb_sz). Other information (age, genre, ...) could be added the same way (Chapter 9).

    class CollabNN(Module):
        def __init__(self, user_sz, item_sz, y_range=(0,5.5), n_act=100):
            self.user_factors = Embedding(*user_sz)
            self.item_factors = Embedding(*item_sz)
            self.layers = nn.Sequential(nn.Linear(user_sz[1]+item_sz[1], n_act), nn.ReLU(),
                                        nn.Linear(n_act, 1))
            self.y_range = y_range
        def forward(self, x):
            embs = self.user_factors(x[:,0]),self.item_factors(x[:,1])
            x = self.layers(torch.cat(embs, dim=1))
            return sigmoid_range(x, *self.y_range)

    learn = collab_learner(dls, use_nn=True, y_range=(0, 5.5), layers=[100,50])

Run:
    python 07_bootstrapping_and_deep_learning_collab.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from common import Embedding, Learner, MSELossFlat, set_seed, sigmoid_range  # noqa: E402
from common.collab import collab_learner  # noqa: E402
from common.tabular import emb_sz_rule  # noqa: E402
from movielens_data import get_dls, ratings  # noqa: E402

dls = get_dls()
n_users, n_movies = len(dls.classes["user"]), len(dls.classes["title"])

# ---- Bootstrapping: a new user gets the average user's factors ---------------------------------------
set_seed(42)
learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))
learn.silent = True
learn.fit_one_cycle(5, 5e-3, wd=0.1)
m = learn.model
avg_user = m.u_weight.weight[1:].mean(0)
avg_bias = m.u_bias.weight[1:].mean()
scores = sigmoid_range(m.i_weight.weight @ avg_user + m.i_bias.weight.squeeze() + avg_bias, 0, 5.5).detach().cpu()
popular = set(ratings.title.value_counts().index[:300])
best = [dls.classes["title"][i] for i in scores.argsort(descending=True) if dls.classes["title"][i] in popular][:5]
print("Recommendations for a brand-new user (average user, popular movies only):", best)

# ---- Deep learning for collaborative filtering --------------------------------------------------------------
embs = [(n_users, emb_sz_rule(n_users)), (n_movies, emb_sz_rule(n_movies))]
print("\nembedding sizes from fastai's rule of thumb (get_emb_sz):", embs)


class CollabNN(nn.Module):
    def __init__(self, user_sz, item_sz, y_range=(0, 5.5), n_act=100):
        super().__init__()
        self.user_factors = Embedding(*user_sz)
        self.item_factors = Embedding(*item_sz)
        self.layers = nn.Sequential(nn.Linear(user_sz[1] + item_sz[1], n_act), nn.ReLU(), nn.Linear(n_act, 1))
        self.y_range = y_range

    def forward(self, x):
        embs = self.user_factors(x[:, 0]), self.item_factors(x[:, 1])
        x = self.layers(torch.cat(embs, dim=1))
        return sigmoid_range(x, *self.y_range)


print("\nCollabNN:")
set_seed(42)
learn = Learner(dls, CollabNN(*embs), loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.01)

print("\ncollab_learner(use_nn=True, layers=[100,50]):")
set_seed(42)
learn = collab_learner(dls, use_nn=True, y_range=(0, 5.5), layers=[100, 50])
learn.fit_one_cycle(5, 5e-3, wd=0.1)
print(learn.model)
