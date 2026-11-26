"""
Chapter 8 — Collaborative Filtering Deep Dive
Topic 5: Creating our own Embedding module (book section of the same name)

What is nn.Embedding underneath? A matrix of parameters, initialised randomly, that we index into.
To make PyTorch treat a tensor as a parameter of a module (so that .parameters() returns it and
the optimizer updates it), wrap it in nn.Parameter:

    class T(Module):
        def __init__(self): self.a = torch.ones(3)          # NOT a parameter
    class T(Module):
        def __init__(self): self.a = nn.Parameter(torch.ones(3))   # a parameter

    def create_params(size):
        return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))

Run:
    python 05_own_embedding_module.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from common import Learner, MSELossFlat, set_seed, sigmoid_range  # noqa: E402
from movielens_data import get_dls  # noqa: E402


class T1(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.ones(3)


class T2(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(3))


class T3(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(1, 3, bias=False)


print("plain tensor:     parameters =", list(T1().parameters()))
print("nn.Parameter:     parameters =", list(T2().parameters()))
print("nn.Linear inside: parameters =", list(T3().parameters()), "| its weight's type:", type(T3().a.weight).__name__)


def create_params(size):
    return nn.Parameter(torch.zeros(*size).normal_(0, 0.01))


class DotProductBias(nn.Module):
    """The same model as before, with no nn.Embedding: index directly into parameter matrices."""

    def __init__(self, n_users, n_movies, n_factors, y_range=(0, 5.5)):
        super().__init__()
        self.user_factors = create_params([n_users, n_factors])
        self.user_bias = create_params([n_users])
        self.movie_factors = create_params([n_movies, n_factors])
        self.movie_bias = create_params([n_movies])
        self.y_range = y_range

    def forward(self, x):
        users = self.user_factors[x[:, 0]]
        movies = self.movie_factors[x[:, 1]]
        res = (users * movies).sum(dim=1)
        res = res + self.user_bias[x[:, 0]] + self.movie_bias[x[:, 1]]
        return sigmoid_range(res, *self.y_range)


set_seed(42)
dls = get_dls()
model = DotProductBias(len(dls.classes["user"]), len(dls.classes["title"]), 50)
print("\nour model's parameters:", [(n, tuple(p.shape)) for n, p in model.named_parameters()])
learn = Learner(dls, model, loss_func=MSELossFlat())
learn.fit_one_cycle(5, 5e-3, wd=0.1)
