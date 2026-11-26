"""
Chapter 8 — Collaborative Filtering Deep Dive
Topic 3: Collaborative filtering from scratch (book section of the same name)

A PyTorch model is a class inheriting from nn.Module, with a `forward` method. The embeddings use
fastai's Embedding (common.Embedding): an nn.Embedding initialised with small random numbers. Three versions:
  1. DotProduct: rating = user factors . movie factors
  2. + y_range: force predictions into (0, 5.5) with sigmoid_range (the top is a bit above 5 because
     a sigmoid never reaches its limit)
  3. DotProductBias: + a bias per user (some people rate everything high) and per movie (some movies
     are just good)

    class DotProductBias(Module):
        def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
            self.user_factors = Embedding(n_users, n_factors)
            self.user_bias = Embedding(n_users, 1)
            self.movie_factors = Embedding(n_movies, n_factors)
            self.movie_bias = Embedding(n_movies, 1)
            self.y_range = y_range
        def forward(self, x):
            users = self.user_factors(x[:,0])
            movies = self.movie_factors(x[:,1])
            res = (users * movies).sum(dim=1, keepdim=True)
            res += self.user_bias(x[:,0]) + self.movie_bias(x[:,1])
            return sigmoid_range(res, *self.y_range)

Run:
    python 03_collab_from_scratch.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch import nn  # noqa: E402

from common import Embedding, Learner, MSELossFlat, set_seed, sigmoid_range  # noqa: E402
from movielens_data import get_dls  # noqa: E402


# Python classes in one line: __init__ runs when you create the object, methods take `self` first
class Example:
    def __init__(self, a):
        self.a = a

    def say(self, x):
        return f"Hello {self.a}, {x}."


print(Example("Sylvain").say("nice to meet you"))


class DotProduct(nn.Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=None):
        super().__init__()
        self.user_factors = Embedding(n_users, n_factors)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.y_range = y_range

    def forward(self, x):
        users = self.user_factors(x[:, 0])
        movies = self.movie_factors(x[:, 1])
        res = (users * movies).sum(dim=1)
        return res if self.y_range is None else sigmoid_range(res, *self.y_range)


class DotProductBias(nn.Module):
    def __init__(self, n_users, n_movies, n_factors, y_range=(0, 5.5)):
        super().__init__()
        self.user_factors = Embedding(n_users, n_factors)
        self.user_bias = Embedding(n_users, 1)
        self.movie_factors = Embedding(n_movies, n_factors)
        self.movie_bias = Embedding(n_movies, 1)
        self.y_range = y_range

    def forward(self, x):
        users = self.user_factors(x[:, 0])
        movies = self.movie_factors(x[:, 1])
        res = (users * movies).sum(dim=1, keepdim=True)
        res = res + self.user_bias(x[:, 0]) + self.movie_bias(x[:, 1])
        return sigmoid_range(res, *self.y_range)


dls = get_dls()
n_users, n_movies = len(dls.classes["user"]), len(dls.classes["title"])

for name, model in [("1. DotProduct", DotProduct(n_users, n_movies, 50)),
                    ("2. DotProduct with y_range=(0,5.5)", DotProduct(n_users, n_movies, 50, y_range=(0, 5.5))),
                    ("3. DotProductBias", DotProductBias(n_users, n_movies, 50))]:
    print(f"\n{name}")
    set_seed(42)
    learn = Learner(dls, model, loss_func=MSELossFlat())
    learn.fit_one_cycle(5, 5e-3)
print("\nThe bias version gets the best training loss, but its validation loss gets worse at the end:"
      " overfitting. Next: weight decay.")
