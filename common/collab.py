"""Collaborative filtering (Ch. 1 and 8): DataLoaders of (user, item) -> rating, the dot-product
model with biases (EmbeddingDotBias) and the neural-net version (EmbeddingNN)."""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .learner import DataLoaders, Learner, MSELossFlat
from .models import sigmoid_range
from .tabular import TabularModel, emb_sz_rule
from .utils import get_device


class CollabDataLoaders(DataLoaders):
    @classmethod
    def from_df(cls, ratings, user_name=None, item_name=None, rating_name=None, valid_pct=0.2, seed=None, bs=64):
        """By default the first three columns are user, item and rating. Users and items are replaced by
        indices into `dls.classes[user_name]` / `dls.classes[item_name]` (index 0 = '#na#')."""
        user_name = user_name or ratings.columns[0]
        item_name = item_name or ratings.columns[1]
        rating_name = rating_name or ratings.columns[2]
        classes = {c: ["#na#"] + sorted(ratings[c].unique().tolist()) for c in (user_name, item_name)}
        maps = {c: {v: i for i, v in enumerate(classes[c])} for c in classes}
        x = torch.tensor(np.stack([ratings[user_name].map(maps[user_name]).values,
                                   ratings[item_name].map(maps[item_name]).values], 1), dtype=torch.long)
        y = torch.tensor(ratings[rating_name].values, dtype=torch.float).view(-1, 1)
        idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed) if seed is not None else None)
        cut = int(valid_pct * len(x))
        tr, va = TensorDataset(x[idx[cut:]], y[idx[cut:]]), TensorDataset(x[idx[:cut]], y[idx[:cut]])
        dls = cls(DataLoader(tr, bs, shuffle=True, drop_last=True), DataLoader(va, bs * 2),
                  loss_func=MSELossFlat(), device=get_device())
        dls.classes, dls.user_name, dls.item_name, dls.rating_name = classes, user_name, item_name, rating_name
        return dls

    @classmethod
    def from_csv(cls, csv, **kwargs):
        return cls.from_df(pd.read_csv(csv), **kwargs)

    def show_batch(self, max_n=10):
        xb, yb = self.train.one_batch()
        rows = [(self.classes[self.user_name][int(u)], self.classes[self.item_name][int(i)], float(r))
                for (u, i), r in zip(xb[:max_n].cpu(), yb[:max_n, 0].cpu())]
        print(pd.DataFrame(rows, columns=[self.user_name, self.item_name, self.rating_name]).to_string())


class EmbeddingDotBias(nn.Module):
    """rating = sigmoid_range(user_factors . item_factors + user_bias + item_bias, *y_range)"""

    def __init__(self, n_factors, n_users, n_items, y_range=None):
        super().__init__()
        self.u_weight, self.i_weight = nn.Embedding(n_users, n_factors), nn.Embedding(n_items, n_factors)
        self.u_bias, self.i_bias = nn.Embedding(n_users, 1), nn.Embedding(n_items, 1)
        for e in (self.u_weight, self.i_weight):
            nn.init.normal_(e.weight, 0, 0.01)
        for e in (self.u_bias, self.i_bias):
            nn.init.zeros_(e.weight)
        self.y_range = y_range

    def forward(self, x):
        users, items = x[:, 0], x[:, 1]
        dot = (self.u_weight(users) * self.i_weight(items)).sum(1)
        res = dot + self.u_bias(users).squeeze(1) + self.i_bias(items).squeeze(1)
        return res if self.y_range is None else sigmoid_range(res, *self.y_range)


class EmbeddingNN(TabularModel):
    """Concatenate the user and item embeddings and feed them through an MLP (Ch. 8)."""

    def __init__(self, emb_szs, layers, **kwargs):
        super().__init__(emb_szs, n_cont=0, out_sz=1, layers=layers, **kwargs)

    def forward(self, x):
        return super().forward(x)


def collab_learner(dls, n_factors=50, use_nn=False, emb_szs=None, layers=None, y_range=None, loss_func=None, **kwargs):
    n_users, n_items = len(dls.classes[dls.user_name]), len(dls.classes[dls.item_name])
    if use_nn:
        emb = [(n_users, emb_sz_rule(n_users)), (n_items, emb_sz_rule(n_items))] if emb_szs is None else emb_szs
        model = EmbeddingNN(emb, layers or [100], y_range=y_range)
    else:
        model = EmbeddingDotBias(n_factors, n_users, n_items, y_range=y_range)
    return Learner(dls, model, loss_func=loss_func or MSELossFlat(), **kwargs)
