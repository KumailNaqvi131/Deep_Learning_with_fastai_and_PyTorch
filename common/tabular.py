"""Tabular data the fastai way (Ch. 1, 9 and 15): add_datepart, TabularPandas with the procs
Categorify / FillMissing / Normalize, and a neural net with categorical embeddings (TabularModel)."""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .learner import CrossEntropyLossFlat, DataLoaders, Learner, MSELossFlat
from .models import Embedding, SigmoidRange
from .utils import get_device


# ==================================================================================================
# Feature engineering
# ==================================================================================================
def add_datepart(df, field_name, drop=True, time=False):
    """Replace a date column by many useful columns: saleYear, saleMonth, saleDayofweek, saleIs_month_end,
    ..., saleElapsed (seconds since 1970). Trees and nets can't use a raw date, but they can use these."""
    field = df[field_name]
    if not pd.api.types.is_datetime64_any_dtype(field):
        df[field_name] = field = pd.to_datetime(field, format="mixed")
    prefix = field_name[:-4] if field_name.lower().endswith("date") else field_name
    attrs = ["Year", "Month", "Week", "Day", "Dayofweek", "Dayofyear", "Is_month_end", "Is_month_start",
             "Is_quarter_end", "Is_quarter_start", "Is_year_end", "Is_year_start"]
    if time:
        attrs += ["Hour", "Minute", "Second"]
    for n in attrs:
        if n == "Week":
            df[prefix + n] = field.dt.isocalendar().week.astype(field.dt.day.dtype)
        else:
            df[prefix + n] = getattr(field.dt, n.lower())       # e.g. field.dt.dayofweek, field.dt.is_month_end
    df[prefix + "Elapsed"] = (field - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("int64")
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    return df


def cont_cat_split(df, max_card=20, dep_var=None):
    """Integer columns with more than max_card values and float columns are continuous, the rest categorical."""
    cont, cat = [], []
    for col in df.columns:
        if col == dep_var or (isinstance(dep_var, (list, tuple)) and col in dep_var):
            continue
        if (pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() > max_card) or pd.api.types.is_float_dtype(df[col]):
            cont.append(col)
        else:
            cat.append(col)
    return cont, cat


# ==================================================================================================
# Procs: transforms set up on the training set, then applied to every row
# ==================================================================================================
class Categorify:
    """Category values -> integer codes. 0 is reserved for missing/unknown ('#na#')."""
    order = 1

    def setup(self, to, train_idx):
        self.classes = {}
        for c in to.cat_names:
            vals = to.items.loc[train_idx, c]
            cats = pd.Categorical(vals.dropna()).categories if not isinstance(vals.dtype, pd.CategoricalDtype) \
                else vals.cat.categories
            self.classes[c] = ["#na#"] + list(cats)

    def __call__(self, to):
        for c in to.cat_names:
            to.items[c] = pd.Categorical(to.items[c], categories=self.classes[c][1:],
                                         ordered=isinstance(to.items[c].dtype, pd.CategoricalDtype) and to.items[c].cat.ordered).codes + 1
        to.classes = self.classes


class FillMissing:
    """Missing continuous values -> the training median, plus a boolean column <name>_na."""
    order = 0

    def setup(self, to, train_idx):
        self.na_dict = {c: to.items.loc[train_idx, c].median() for c in to.cont_names
                        if to.items.loc[train_idx, c].isna().any()}
        for c in self.na_dict:
            if f"{c}_na" not in to.cat_names:
                to.cat_names.append(f"{c}_na")

    def __call__(self, to):
        for c, med in self.na_dict.items():
            to.items[f"{c}_na"] = to.items[c].isna()
            to.items[c] = to.items[c].fillna(med)


class Normalize:
    """Continuous columns -> (x - mean) / std using training-set statistics (only needed for neural nets)."""
    order = 2

    def setup(self, to, train_idx):
        self.means = {c: to.items.loc[train_idx, c].mean() for c in to.cont_names}
        self.stds = {c: to.items.loc[train_idx, c].std() + 1e-7 for c in to.cont_names}

    def __call__(self, to):
        for c in to.cont_names:
            to.items[c] = (to.items[c] - self.means[c]) / self.stds[c]


class TabularPandas:
    """A DataFrame processed by `procs`, split into train/valid. `to.train.xs`, `to.train.y`, ...

        to = TabularPandas(df, procs=[Categorify, FillMissing], cat_names=cat, cont_names=cont,
                           y_names='SalePrice', splits=splits)
    """

    def __init__(self, df, procs=None, cat_names=None, cont_names=None, y_names=None, splits=None,
                 y_block=None, reduce_memory=True, do_setup=True):
        self.items = df.copy()
        self.cat_names, self.cont_names = list(cat_names or []), list(cont_names or [])
        self.y_names = [y_names] if isinstance(y_names, str) else list(y_names or [])
        self.procs = sorted([p() if isinstance(p, type) else p for p in (procs or [])], key=lambda p: p.order)
        if splits is None:
            splits = (list(range(len(df))), [])
        idx = self.items.index
        self.splits = (list(idx[list(splits[0])]), list(idx[list(splits[1])]))
        self.is_classification = y_block == "category" or (
            y_block is None and self.y_names and not pd.api.types.is_numeric_dtype(df[self.y_names[0]]))
        if do_setup:
            for p in self.procs:
                p.setup(self, self.splits[0])
                p(self)
            if self.is_classification:
                self.y_vocab = sorted(self.items.loc[self.splits[0], self.y_names[0]].unique())
                self.items[self.y_names[0]] = self.items[self.y_names[0]].map({v: i for i, v in enumerate(self.y_vocab)})
        self.classes = getattr(self, "classes", {})

    def subset(self, i):
        return _TabSubset(self, self.splits[i])

    train = property(lambda self: self.subset(0))
    valid = property(lambda self: self.subset(1))

    @property
    def xs(self):
        return self.items[self.cat_names + self.cont_names]

    @property
    def y(self):
        return self.items[self.y_names[0]]

    def new(self, df):
        """Process new data (e.g. a test set) with the procs already set up on the training set."""
        new = TabularPandas.__new__(TabularPandas)
        new.items, new.cat_names, new.cont_names = df.copy(), list(self.cat_names), list(self.cont_names)
        new.y_names, new.procs, new.splits = [], self.procs, (list(df.index), [])
        new.is_classification = self.is_classification
        for p in self.procs:
            p(new)
        new.classes = self.classes
        return new

    def show(self, max_n=5):
        df = self.items.loc[self.splits[0][:max_n], self.cat_names + self.cont_names + self.y_names].copy()
        for c in self.cat_names:
            if c in self.classes:
                df[c] = [self.classes[c][int(v)] for v in df[c]]
        print(df.to_string())

    def dataloaders(self, bs=64, device=None):
        device = device or get_device()

        def ds(sub):
            xc = torch.tensor(sub.xs[self.cat_names].values.astype(np.int64)) if self.cat_names else torch.zeros(len(sub.xs), 0, dtype=torch.long)
            xn = torch.tensor(sub.xs[self.cont_names].values.astype(np.float32)) if self.cont_names else torch.zeros(len(sub.xs), 0)
            y = torch.tensor(sub.y.values.astype(np.int64 if self.is_classification else np.float32))
            return TensorDataset(xc, xn, y if self.is_classification else y.view(-1, 1))

        tr, va = ds(self.train), ds(self.valid)
        dls = DataLoaders(DataLoader(tr, bs, shuffle=True, drop_last=len(tr) > bs), DataLoader(va, bs * 2),
                          vocab=getattr(self, "y_vocab", None), c=len(self.y_vocab) if self.is_classification else 1,
                          loss_func=CrossEntropyLossFlat() if self.is_classification else MSELossFlat(),
                          device=device, n_inp=2)
        dls.to, dls.cat_names, dls.cont_names = self, self.cat_names, self.cont_names
        return dls


class _TabSubset:
    def __init__(self, to, idx):
        self.to, self.idx = to, idx

    xs = property(lambda self: self.to.items.loc[self.idx, self.to.cat_names + self.to.cont_names])
    y = property(lambda self: self.to.items.loc[self.idx, self.to.y_names[0]])

    def __len__(self):
        return len(self.idx)


class TabularDataLoaders:
    @staticmethod
    def from_df(df, y_names, cat_names=None, cont_names=None, procs=(Categorify, FillMissing, Normalize),
                valid_pct=0.2, seed=42, bs=64, **kwargs):
        idx = np.random.default_rng(seed).permutation(len(df))
        cut = int(len(df) * valid_pct)
        to = TabularPandas(df, list(procs), cat_names, cont_names, y_names,
                           splits=(sorted(idx[cut:]), sorted(idx[:cut])))
        return to.dataloaders(bs=bs)

    @staticmethod
    def from_csv(csv, path=None, y_names=None, cat_names=None, cont_names=None, procs=(Categorify, FillMissing, Normalize), **kwargs):
        return TabularDataLoaders.from_df(pd.read_csv(csv), y_names, cat_names, cont_names, procs, **kwargs)


# ==================================================================================================
# The model
# ==================================================================================================
def emb_sz_rule(n_cat):
    """fastai's rule of thumb for the size of an embedding with n_cat categories."""
    return min(600, round(1.6 * n_cat ** 0.56))


def get_emb_sz(to, sz_dict=None):
    return [(len(to.classes[c]), (sz_dict or {}).get(c, emb_sz_rule(len(to.classes[c])))) for c in to.cat_names]


class TabularModel(nn.Module):
    """Embeddings for categorical columns + BatchNorm'd continuous columns -> MLP (Ch. 9 and 15)."""

    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None, embed_p=0.0, y_range=None, use_bn=True, bn_final=False):
        super().__init__()
        ps = ps if ps is not None else [0.0] * len(layers)
        ps = ps if isinstance(ps, (list, tuple)) else [ps] * len(layers)
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont) if n_cont else None
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb, self.n_cont = n_emb, n_cont
        sizes = [n_emb + n_cont] + list(layers)
        mods = []
        for i, (ni, no) in enumerate(zip(sizes[:-1], sizes[1:])):
            mods += [nn.Linear(ni, no, bias=not use_bn), nn.ReLU(inplace=True)]
            if use_bn:
                mods.append(nn.BatchNorm1d(no))
            if ps[i]:
                mods.append(nn.Dropout(ps[i]))
        mods.append(nn.Linear(sizes[-1], out_sz))
        if bn_final:
            mods.append(nn.BatchNorm1d(out_sz))
        if y_range is not None:
            mods.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*mods)

    def forward(self, x_cat, x_cont=None):
        if self.n_emb:
            x = torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embeds)], 1)
            x = self.emb_drop(x)
        if self.n_cont:
            xc = self.bn_cont(x_cont) if self.bn_cont is not None else x_cont
            x = torch.cat([x, xc], 1) if self.n_emb else xc
        return self.layers(x)


def tabular_learner(dls, layers=None, emb_szs=None, config=None, n_out=None, y_range=None, loss_func=None,
                    metrics=None, **kwargs):
    layers = [200, 100] if layers is None else layers
    to = dls.to
    emb = get_emb_sz(to, emb_szs)
    model = TabularModel(emb, len(to.cont_names), n_out or dls.c, layers, y_range=y_range, **(config or {}))
    return Learner(dls, model, loss_func=loss_func, metrics=metrics, **kwargs)
