"""
Chapter 8 — Collaborative Filtering Deep Dive
Topic 2: Creating the DataLoaders; embeddings (book section "Creating the DataLoaders")

To look up user 3's factors we could multiply a one-hot vector by the matrix of all users'
factors: the result is row 3. Since multiplying by a one-hot vector is just indexing, PyTorch
provides a layer that does the indexing directly, but whose gradient behaves like the matrix
product: nn.Embedding. The matrix it indexes into is an *embedding matrix*.

    user_factors.t() @ one_hot(3, n_users).float()     ==     user_factors[3]

Run:
    python 02_dataloaders_and_embeddings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from movielens_data import get_dls  # noqa: E402

dls = get_dls()
dls.show_batch(8)
n_users = len(dls.classes["user"])
n_movies = len(dls.classes["title"])
print(f"\nn_users={n_users}, n_movies={n_movies} (each includes '#na#' at index 0)")
x, y = dls.one_batch()
print("x:", tuple(x.shape), "(user index, movie index) | y:", tuple(y.shape))

n_factors = 5
user_factors = torch.randn(n_users, n_factors)
movie_factors = torch.randn(n_movies, n_factors)

one_hot_3 = F.one_hot(torch.tensor(3), n_users).float()
print("\nuser_factors.t() @ one_hot_3:", user_factors.t() @ one_hot_3)
print("user_factors[3]:            ", user_factors[3])

emb = nn.Embedding(n_users, n_factors)
print("\nnn.Embedding does the indexing: emb(tensor([3])) == emb.weight[3]:",
      torch.equal(emb(torch.tensor([3]))[0], emb.weight[3]))
