"""fastai's mid-level data API (Chapter 11), small and readable:

    Transform   an object with encodes (and optionally setups and decodes); applied to a tuple, it
                transforms each element; it only applies to inputs of the type in its annotation
    Pipeline    several transforms composed (and decoded in reverse order)
    TfmdLists   a list of items + a Pipeline, set up on the training items, lazily applied
    Datasets    several TfmdLists in parallel (e.g. one pipeline for x, one for y)
"""

import inspect
import random

import torch
from torch.utils.data import DataLoader

from .learner import DataLoaders
from .utils import get_device


def _annotation(f):
    """The type annotation of a function's first (non-self) parameter, or None."""
    try:
        params = [p for p in inspect.signature(f).parameters.values() if p.name != "self"]
    except (TypeError, ValueError):
        return None
    if not params or params[0].annotation is inspect.Parameter.empty:
        return None
    return params[0].annotation


class Transform:
    """Subclass and define encodes(self, x) / decodes(self, x) / setups(self, items), or wrap a function:
    Transform(f). Type dispatch: if encodes is annotated (def encodes(self, x: int)), other types pass
    through unchanged. Tuples are transformed element by element."""

    def __init__(self, enc=None, dec=None):
        if enc is not None:
            self.encodes = enc
        if dec is not None:
            self.decodes = dec
        self.name = getattr(enc, "__name__", type(self).__name__)

    def setup(self, items=None, train_setup=True):
        if hasattr(self, "setups"):
            self.setups(items)
        return self

    def _call(self, fn_name, x):
        f = getattr(self, fn_name, None)
        if f is None:
            return x
        if isinstance(x, tuple) and not getattr(self, "whole_tuple", False):
            return tuple(self._call(fn_name, o) for o in x)     # tuples: each element on its own
        t = _annotation(f)
        if t is not None and not isinstance(x, t):
            return x                                           # not our type: leave it alone
        return f(x)

    def __call__(self, x):
        return self._call("encodes", x)

    def decode(self, x):
        return self._call("decodes", x)

    def __repr__(self):
        return f"{self.name}: encodes{' + decodes' if hasattr(self, 'decodes') else ''}"


class Pipeline:
    """Compose transforms: Pipeline([tok, num])(text) == num(tok(text)); decode runs them backwards."""

    def __init__(self, funcs=None):
        self.fs = [f if isinstance(f, Transform) else (f() if isinstance(f, type) else Transform(f)) for f in (funcs or [])]

    def setup(self, items=None):
        """Set up each transform on the items *as transformed by the previous ones*."""
        for i, f in enumerate(self.fs):
            if hasattr(f, "setups") and items is not None:
                f.setup([self._apply(o, self.fs[:i]) for o in items])

    @staticmethod
    def _apply(x, fs):
        for f in fs:
            x = f(x)
        return x

    def __call__(self, x):
        return self._apply(x, self.fs)

    def decode(self, x):
        for f in reversed(self.fs):
            x = f.decode(x)
        return x

    def __repr__(self):
        return "Pipeline: " + " -> ".join(f.name for f in self.fs)


class TfmdLists:
    """items + a Pipeline. Indexing applies the pipeline lazily. splits=(train_idx, valid_idx); the
    pipeline is set up on the training items only (e.g. the vocab comes from the training set)."""

    def __init__(self, items, tfms, splits=None, do_setup=True):
        self.items = list(items)
        self.tfms = tfms if isinstance(tfms, Pipeline) else Pipeline(tfms if isinstance(tfms, (list, tuple)) else [tfms])
        self.splits = splits or (list(range(len(self.items))), [])
        if do_setup:
            self.tfms.setup([self.items[i] for i in self.splits[0]])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.tfms(self.items[i])

    def decode(self, x):
        return self.tfms.decode(x)

    def subset(self, i):
        new = TfmdLists([self.items[j] for j in self.splits[i]], self.tfms, do_setup=False)
        return new

    train = property(lambda self: self.subset(0))
    valid = property(lambda self: self.subset(1))

    def dataloaders(self, bs=64, after_item=None, before_batch=None, after_batch=None, device=None, shuffle=True, n_inp=1):
        return _make_dls(self.train, self.valid, bs, after_item, before_batch, after_batch, device, shuffle, n_inp)


class Datasets:
    """Several pipelines on the same items: dsets[i] == (x_pipeline(item), y_pipeline(item))."""

    def __init__(self, items, tfms, splits=None, do_setup=True):
        self.items = list(items)
        self.splits = splits or (list(range(len(self.items))), [])
        self.tls = [TfmdLists(self.items, t, self.splits, do_setup) for t in tfms]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return tuple(tl[i] for tl in self.tls)

    def decode(self, x):
        return tuple(tl.decode(o) for tl, o in zip(self.tls, x))

    def subset(self, i):
        new = Datasets.__new__(Datasets)
        new.items = [self.items[j] for j in self.splits[i]]
        new.splits = (list(range(len(new.items))), [])
        new.tls = [t.__class__(new.items, t.tfms, do_setup=False) for t in self.tls]
        return new

    train = property(lambda self: self.subset(0))
    valid = property(lambda self: self.subset(1))

    def dataloaders(self, bs=64, after_item=None, before_batch=None, after_batch=None, device=None, shuffle=True, n_inp=1):
        return _make_dls(self.train, self.valid, bs, after_item, before_batch, after_batch, device, shuffle, n_inp)


def _make_dls(train, valid, bs, after_item, before_batch, after_batch, device, shuffle, n_inp):
    """DataLoaders with fastai's three hooks: after_item (per item), before_batch (on the list of items,
    e.g. padding), after_batch (on the collated batch, on the device)."""
    device = device or get_device()
    item_pipe = Pipeline(after_item or [])

    def collate(samples):
        samples = [item_pipe(s) for s in samples]
        if before_batch is not None:
            samples = before_batch(samples)
        return torch.utils.data.default_collate(samples)

    batch_pipe = Pipeline(after_batch or [])

    class _DL:
        def __init__(self, ds, shuffle_):
            self.dl = DataLoader(ds, batch_size=bs, shuffle=shuffle_, collate_fn=collate)
            self.dataset = ds

        def __iter__(self):
            for b in self.dl:
                yield batch_pipe(tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in b))

        def __len__(self):
            return len(self.dl)

    from .learner import DeviceDataLoader
    wrap = lambda ds, sh: DeviceDataLoader(_DL(ds, sh), device, train=sh, n_inp=n_inp)  # noqa: E731
    dls = DataLoaders(wrap(train, shuffle), wrap(valid, False), device=device, n_inp=n_inp)
    return dls


class Categorize(Transform):
    """label -> index in a sorted vocab built from the training labels."""

    def setups(self, items):
        self.vocab = sorted(set(items))
        self.o2i = {v: i for i, v in enumerate(self.vocab)}

    def encodes(self, x):
        return torch.tensor(self.o2i[x])

    def decodes(self, x):
        return self.vocab[int(x)]


def pad_input(samples, pad_idx=1):
    """before_batch: pad the token sequences of a list of (x, y) samples to the same length (padding first)."""
    max_len = max(len(s[0]) for s in samples)
    return [(torch.cat([torch.full((max_len - len(s[0]),), pad_idx, dtype=s[0].dtype), s[0]]),) + tuple(s[1:])
            for s in samples]


def random_choice(seq):
    return seq[random.randrange(len(seq))]
