"""The data block API for images, re-implemented in plain PyTorch with fastai's names:

    pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                     get_items=get_image_files,
                     splitter=RandomSplitter(seed=42),
                     get_y=using_attr(RegexLabeller(r'(.+)_\\d+.jpg$'), 'name'),
                     item_tfms=Resize(460),
                     batch_tfms=aug_transforms(size=224, min_scale=0.75))
    dls = pets.dataloaders(path/"images")
    dls.show_batch(fname="batch.png")

How a sample flows (Ch. 2, 5 and 11):
  get_items(source) -> split into train/valid -> get_x / get_y -> blocks open the file and encode
  the label -> item transforms (CPU, one item at a time, e.g. Resize) -> collate into a batch ->
  batch transforms (GPU, whole batch: augmentation, Normalize).
"""

import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from .learner import (BCEWithLogitsLossFlat, CrossEntropyLossFlat, DataLoaders, DeviceDataLoader,
                      Learner, MSELossFlat)
from .utils import get_device, show_images
from .utils import num_workers as default_num_workers

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


# ==================================================================================================
# Getting items and labels
# ==================================================================================================
def get_files(path, extensions=None, recurse=True, folders=None):
    path = Path(path)
    exts = {e.lower() for e in extensions} if extensions else None
    roots = [path / f for f in folders] if folders else [path]
    files = []
    for root in roots:
        it = root.rglob("*") if recurse else root.glob("*")
        files += [p for p in it if p.is_file() and not p.name.startswith(".")
                  and (exts is None or p.suffix.lower() in exts)]
    return sorted(files)


def get_image_files(path, recurse=True, folders=None):
    """All image files under `path` (sorted, so results are repeatable)."""
    return get_files(path, IMAGE_EXTENSIONS, recurse, folders)


def verify_images(fns):
    """Return the files that can't be opened as images (Ch. 2: downloads often contain broken files)."""
    bad = []
    for f in fns:
        try:
            with Image.open(f) as im:
                im.load()
        except Exception:
            bad.append(f)
    return bad


def parent_label(o):
    """Label = name of the parent folder (bears/grizzly/0001.jpg -> 'grizzly')."""
    return Path(o).parent.name


class RegexLabeller:
    """Label = first group of a regular expression applied to the file name or path."""

    def __init__(self, pat, match=False):
        self.pat = re.compile(pat)
        self.match = match

    def __call__(self, o):
        res = (self.pat.match if self.match else self.pat.search)(str(o))
        assert res, f"Failed to find '{self.pat.pattern}' in '{o}'"
        return res.group(1)


class using_attr:
    """Apply `f` to an attribute: using_attr(labeller, 'name')(path) == labeller(path.name)."""

    def __init__(self, f, attr):
        self.f, self.attr = f, attr

    def __call__(self, o):
        return self.f(getattr(o, self.attr))


class ColReader:
    """Read column(s) of a DataFrame row, optionally adding a prefix/suffix or splitting labels."""

    def __init__(self, cols, pref="", suff="", label_delim=None):
        self.cols, self.pref, self.suff, self.label_delim = cols, pref, suff, label_delim

    def __call__(self, row):
        v = row[self.cols]
        if self.label_delim is not None:
            return v.split(self.label_delim) if isinstance(v, str) else []
        if isinstance(self.pref, Path):                 # pref=path/'train' -> path/'train'/<value>
            return self.pref / f"{v}{self.suff}"
        if self.pref or self.suff:
            return f"{self.pref}{v}{self.suff}"
        return v


# ---- splitters: items -> (train indices, valid indices) ----
class RandomSplitter:
    def __init__(self, valid_pct=0.2, seed=None):
        self.valid_pct, self.seed = valid_pct, seed

    def __call__(self, items):
        g = torch.Generator().manual_seed(self.seed) if self.seed is not None else None
        idx = torch.randperm(len(items), generator=g).tolist()
        cut = int(self.valid_pct * len(items))
        return sorted(idx[cut:]), sorted(idx[:cut])


class GrandparentSplitter:
    """train/valid decided by the grand-parent folder: path/train/cat/1.jpg -> 'train'."""

    def __init__(self, train_name="train", valid_name="valid"):
        self.train_name, self.valid_name = train_name, valid_name

    def __call__(self, items):
        gp = [Path(o).parent.parent.name for o in items]
        return ([i for i, g in enumerate(gp) if g == self.train_name],
                [i for i, g in enumerate(gp) if g == self.valid_name])


class FuncSplitter:
    """Items for which `func(item)` is True go to the validation set."""

    def __init__(self, func):
        self.func = func

    def __call__(self, items):
        flags = [bool(self.func(o)) for o in items]
        return [i for i, f in enumerate(flags) if not f], [i for i, f in enumerate(flags) if f]


class ColSplitter:
    """DataFrame rows with df[col] True go to the validation set."""

    def __init__(self, col="is_valid"):
        self.col = col

    def __call__(self, items):
        flags = [bool(r[self.col]) for r in items]
        return [i for i, f in enumerate(flags) if not f], [i for i, f in enumerate(flags) if f]


class IndexSplitter:
    def __init__(self, valid_idx):
        self.valid_idx = set(int(i) for i in valid_idx)

    def __call__(self, items):
        return [i for i in range(len(items)) if i not in self.valid_idx], sorted(self.valid_idx)


# ==================================================================================================
# Blocks: how to open an input and how to encode a label
# ==================================================================================================
class PILImage:
    mode = "RGB"

    @classmethod
    def create(cls, fn):
        img = fn if isinstance(fn, Image.Image) else Image.open(fn)
        return img.convert(cls.mode)


class PILImageBW(PILImage):
    mode = "L"


class PILMask:
    @staticmethod
    def create(fn):
        img = fn if isinstance(fn, Image.Image) else Image.open(fn)
        return img if img.mode in ("L", "P") else img.convert("L")


class ImageBlock:
    kind = "image"

    def __init__(self, cls=PILImage):
        self.cls = cls

    def create(self, x):
        return self.cls.create(x)

    def setup(self, values):
        pass

    def encode(self, x):
        return x


class CategoryBlock:
    """Single label -> integer index into a sorted `vocab`."""
    kind = "category"
    loss_func = staticmethod(lambda: CrossEntropyLossFlat())

    def __init__(self, vocab=None):
        self.vocab = list(vocab) if vocab is not None else None

    def create(self, x):
        return x

    def setup(self, values):
        if self.vocab is None:
            self.vocab = sorted(set(values))
        self.o2i = {v: i for i, v in enumerate(self.vocab)}

    def encode(self, y):
        return torch.tensor(self.o2i[y])

    def decode(self, y):
        return self.vocab[int(y)]


class MultiCategoryBlock:
    """List of labels -> one-hot encoded float vector (Ch. 6)."""
    kind = "multicat"
    loss_func = staticmethod(lambda: BCEWithLogitsLossFlat())

    def __init__(self, encoded=False, vocab=None):
        self.encoded, self.vocab = encoded, list(vocab) if vocab is not None else None

    def create(self, x):
        return x

    def setup(self, values):
        if self.vocab is None:
            self.vocab = sorted({v for vals in values for v in vals})
        self.o2i = {v: i for i, v in enumerate(self.vocab)}

    def encode(self, ys):
        t = torch.zeros(len(self.vocab))
        if self.encoded:
            return torch.as_tensor(ys, dtype=torch.float)
        for y in ys:
            t[self.o2i[y]] = 1.0
        return t

    def decode(self, t):
        return [self.vocab[i] for i in torch.where(torch.as_tensor(t) > 0.5)[0]]


class PointBlock:
    """Key points (x, y) in pixels. After the item transforms they are rescaled to -1..+1 (Ch. 6)."""
    kind = "point"
    loss_func = staticmethod(lambda: MSELossFlat())
    vocab = None

    def create(self, x):
        return torch.as_tensor(x, dtype=torch.float).view(-1, 2)

    def setup(self, values):
        pass

    def encode(self, y):
        return y


class MaskBlock:
    """Segmentation mask: one class code per pixel (Ch. 1 and 15)."""
    kind = "mask"
    loss_func = staticmethod(lambda: CrossEntropyLossFlat(axis=1))

    def __init__(self, codes=None):
        self.vocab = list(codes) if codes is not None else None

    def create(self, x):
        return PILMask.create(x)

    def setup(self, values):
        pass

    def encode(self, y):
        return y


class RegressionBlock:
    kind = "float"
    loss_func = staticmethod(lambda: MSELossFlat())
    vocab = None

    def create(self, x):
        return x

    def setup(self, values):
        pass

    def encode(self, y):
        return torch.as_tensor(y, dtype=torch.float)


# ==================================================================================================
# Item transforms (run on the CPU, on one sample = [input, target] at a time)
# ==================================================================================================
def _size(size):
    return (size, size) if isinstance(size, int) else tuple(size)      # (height, width)


def _resize_sample(sample, kinds, box, out_hw, pad=None):
    """Crop `box` (left, top, right, bottom; may extend outside the image) and resize it to out_hw.
    Images use bilinear interpolation, masks nearest-neighbour, points are moved accordingly."""
    oh, ow = out_hw
    l, t, r, b = box
    res = []
    for x, k in zip(sample, kinds):
        if k == "image" or k == "mask":
            resample = Image.BILINEAR if k == "image" else Image.NEAREST
            if pad == "reflection" and k == "image" and (l < 0 or t < 0 or r > x.width or b > x.height):
                a = np.asarray(x)
                pw = [(max(0, -int(t)), max(0, int(math.ceil(b)) - x.height)), (max(0, -int(l)), max(0, int(math.ceil(r)) - x.width))]
                a = np.pad(a, pw + ([(0, 0)] if a.ndim == 3 else []), mode="reflect")
                x2 = Image.fromarray(a)
                box2 = (l + pw[1][0], t + pw[0][0], r + pw[1][0], b + pw[0][0])
                res.append(x2.resize((ow, oh), resample, box=box2))
            else:
                res.append(x.crop(tuple(int(round(v)) for v in box)).resize((ow, oh), resample))
        elif k == "point":
            sx, sy = ow / (r - l), oh / (b - t)
            res.append((x - torch.tensor([l, t])) * torch.tensor([sx, sy]))
        else:
            res.append(x)
    return res


def _img_size(sample, kinds):
    for x, k in zip(sample, kinds):
        if k in ("image", "mask"):
            return x.width, x.height
    raise ValueError("no image in sample")


class ResizeMethod:
    Crop, Squish, Pad = "crop", "squish", "pad"


class Resize:
    """Resize every item to `size` (Ch. 2). method: 'crop' (random crop on train, centre crop on valid),
    'squish' (change the aspect ratio), or 'pad' (fit inside, pad with reflection or zeros)."""

    def __init__(self, size, method="crop", pad_mode="reflection"):
        self.size, self.method, self.pad_mode = _size(size), method, pad_mode

    def __call__(self, sample, kinds, train):
        w, h = _img_size(sample, kinds)
        oh, ow = self.size
        if self.method == "squish":
            return _resize_sample(sample, kinds, (0, 0, w, h), (oh, ow))
        if self.method == "crop":
            scale = min(w / ow, h / oh)                 # biggest box with the output aspect ratio
            cw, ch = ow * scale, oh * scale
            px, py = (random.random(), random.random()) if train else (0.5, 0.5)
            l, t = (w - cw) * px, (h - ch) * py
            return _resize_sample(sample, kinds, (l, t, l + cw, t + ch), (oh, ow))
        scale = max(w / ow, h / oh)                     # 'pad': smallest box that contains the image
        cw, ch = ow * scale, oh * scale
        l, t = (w - cw) / 2, (h - ch) / 2
        return _resize_sample(sample, kinds, (l, t, l + cw, t + ch), (oh, ow),
                              pad="reflection" if self.pad_mode == "reflection" else None)

    def __repr__(self):
        return f"Resize({self.size}, method='{self.method}')"


class RandomResizedCrop:
    """Train: a random crop covering between min_scale and 100% of the area, with a random aspect
    ratio, resized to `size`. Valid: a centre crop (Ch. 2: the standard augmentation for images)."""

    def __init__(self, size, min_scale=0.08, ratio=(3 / 4, 4 / 3)):
        self.size, self.min_scale, self.ratio = _size(size), min_scale, ratio

    def __call__(self, sample, kinds, train):
        w, h = _img_size(sample, kinds)
        oh, ow = self.size
        if train:
            for _ in range(10):
                area = random.uniform(self.min_scale, 1.0) * w * h
                r = math.exp(random.uniform(math.log(self.ratio[0]), math.log(self.ratio[1])))
                cw, ch = math.sqrt(area * r), math.sqrt(area / r)
                if cw <= w and ch <= h:
                    l, t = random.uniform(0, w - cw), random.uniform(0, h - ch)
                    return _resize_sample(sample, kinds, (l, t, l + cw, t + ch), (oh, ow))
        scale = min(w / ow, h / oh)
        cw, ch = ow * scale, oh * scale
        return _resize_sample(sample, kinds, ((w - cw) / 2, (h - ch) / 2, (w + cw) / 2, (h + ch) / 2), (oh, ow))


def _to_tensor(sample, kinds):
    out = []
    for x, k in zip(sample, kinds):
        if k == "image":
            a = torch.from_numpy(np.array(x, dtype=np.uint8))
            out.append(a.permute(2, 0, 1).contiguous() if a.ndim == 3 else a.unsqueeze(0))
        elif k == "mask":
            out.append(torch.from_numpy(np.array(x, dtype=np.int64)))
        else:
            out.append(x)
    return out


# ==================================================================================================
# Batch transforms (run on the GPU on a whole batch)
# ==================================================================================================
class IntToFloatTensor:
    """uint8 images 0..255 -> float 0..1 (always the first batch transform)."""

    def __call__(self, b, kinds, train):
        return tuple(x.float() / 255.0 if k == "image" and x.dtype == torch.uint8 else x for x, k in zip(b, kinds))


class Normalize:
    """(x - mean) / std on each channel. `decode` undoes it, for showing images.
    Normalize() without statistics computes them from the first batch it sees (like fastai)."""

    def __init__(self, mean=None, std=None):
        self.mean = torch.tensor(mean).view(1, -1, 1, 1) if mean is not None else None
        self.std = torch.tensor(std).view(1, -1, 1, 1) if std is not None else None

    @classmethod
    def from_stats(cls, mean, std):
        return cls(mean, std)

    def __call__(self, b, kinds, train):
        if self.mean is None:
            x = b[kinds.index("image")].float()
            self.mean = x.mean(dim=(0, 2, 3), keepdim=True).cpu()
            self.std = x.std(dim=(0, 2, 3), keepdim=True).cpu() + 1e-7
        return tuple((x - self.mean.to(x.device)) / self.std.to(x.device) if k == "image" else x
                     for x, k in zip(b, kinds))

    def decode(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


class AffineAndLighting:
    """GPU data augmentation for a whole batch, like fastai's aug_transforms (Ch. 2 and 5).

    Flip, rotation, zoom, a little shear (fastai uses a perspective 'warp'; a shear is similar)
    and an optional random-resized crop are combined into ONE affine matrix per image, so each
    image is interpolated only once (the point of presizing). Masks follow the same geometry
    (nearest neighbour); key points are moved with the inverse transform. Then brightness and
    contrast change. On the validation set only the final resize (a centre crop) is done.
    """

    def __init__(self, size=None, min_scale=1.0, do_flip=True, flip_vert=False, max_rotate=10.0,
                 min_zoom=1.0, max_zoom=1.1, max_warp=0.2, max_lighting=0.2, p_affine=0.75, p_lighting=0.75,
                 mult=1.0, pad_mode="reflection"):
        self.size = _size(size) if size is not None else None
        self.min_scale, self.do_flip, self.flip_vert = min_scale, do_flip, flip_vert
        self.max_rotate, self.min_zoom, self.max_zoom = max_rotate * mult, min_zoom, max_zoom * mult if max_zoom > 1 else max_zoom
        self.max_warp, self.max_lighting = max_warp * mult, max_lighting * mult
        self.p_affine, self.p_lighting, self.pad_mode = p_affine, p_lighting, pad_mode

    def _theta(self, n, h, w, oh, ow, train, device):
        theta = torch.zeros(n, 2, 3, device=device)
        theta[:, 0, 0] = theta[:, 1, 1] = 1.0
        # crop to the output aspect ratio: centred on valid, random size/position on train
        r_in, r_out = w / h, ow / oh
        sx, sy = (r_out / r_in, 1.0) if r_out < r_in else (1.0, r_in / r_out)
        sx, sy = torch.full((n,), sx, device=device), torch.full((n,), sy, device=device)
        tx, ty = torch.zeros(n, device=device), torch.zeros(n, device=device)
        if train and self.min_scale < 1:
            area = torch.empty(n, device=device).uniform_(self.min_scale, 1.0)
            ratio = torch.exp(torch.empty(n, device=device).uniform_(math.log(3 / 4), math.log(4 / 3)))
            kx, ky = (area * ratio).sqrt().clamp(max=1), (area / ratio).sqrt().clamp(max=1)
            sx, sy = sx * kx, sy * ky
            tx, ty = (1 - sx) * (2 * torch.rand(n, device=device) - 1), (1 - sy) * (2 * torch.rand(n, device=device) - 1)
        m = torch.zeros(n, 2, 2, device=device)
        m[:, 0, 0], m[:, 1, 1] = sx, sy
        if train:
            do = torch.rand(n, device=device) < self.p_affine
            ang = torch.empty(n, device=device).uniform_(-self.max_rotate, self.max_rotate) * math.pi / 180 * do
            zoom = torch.where(do, torch.empty(n, device=device).uniform_(self.min_zoom, self.max_zoom), torch.ones(n, device=device))
            shear = torch.empty(n, device=device).uniform_(-self.max_warp, self.max_warp) * do
            rot = torch.stack([torch.stack([ang.cos(), -ang.sin()], -1), torch.stack([ang.sin(), ang.cos()], -1)], 1)
            sh = torch.eye(2, device=device).repeat(n, 1, 1)
            sh[:, 0, 1] = shear
            m = m @ rot @ sh / zoom.view(n, 1, 1)
            if self.do_flip:
                flip = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
                m[:, :, 0] = m[:, :, 0] * flip.view(n, 1)
            if self.flip_vert:
                flip = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
                m[:, :, 1] = m[:, :, 1] * flip.view(n, 1)
        theta[:, :, :2] = m
        theta[:, 0, 2], theta[:, 1, 2] = tx, ty
        return theta

    def __call__(self, b, kinds, train):
        idx = kinds.index("image")
        x = b[idx]
        n, c, h, w = x.shape
        oh, ow = self.size or (h, w)
        if not train and (oh, ow) == (h, w):
            return b
        theta = self._theta(n, h, w, oh, ow, train, x.device)
        grid = F.affine_grid(theta, (n, c, oh, ow), align_corners=False)
        out = []
        for t, k in zip(b, kinds):
            if k == "image":
                t = F.grid_sample(t, grid, mode="bilinear", padding_mode=self.pad_mode, align_corners=False)
                if train and self.max_lighting > 0:
                    do = (torch.rand(n, 1, 1, 1, device=t.device) < self.p_lighting).float()
                    bright = (torch.rand(n, 1, 1, 1, device=t.device) - 0.5) * self.max_lighting * do
                    contr = torch.exp((torch.rand(n, 1, 1, 1, device=t.device) * 2 - 1) * math.log(1 + self.max_lighting) * do)
                    t = ((t - 0.5) * contr + 0.5 + bright).clamp(0, 1)
            elif k == "mask":
                t = F.grid_sample(t.unsqueeze(1).float(), grid, mode="nearest", padding_mode=self.pad_mode,
                                  align_corners=False).squeeze(1).long()
            elif k == "point":                          # p_in = A p_out + t  =>  p_out = A^-1 (p_in - t)
                a_inv = torch.linalg.inv(theta[:, :, :2])
                t = ((t - theta[:, None, :, 2]) @ a_inv.transpose(1, 2))
            out.append(t)
        return tuple(out)


def aug_transforms(mult=1.0, do_flip=True, flip_vert=False, max_rotate=10.0, min_zoom=1.0, max_zoom=1.1,
                   max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75, size=None, min_scale=1.0,
                   pad_mode="reflection"):
    """fastai's standard set of augmentations (same names and defaults), as a list of batch transforms."""
    return [AffineAndLighting(size=size, min_scale=min_scale, do_flip=do_flip, flip_vert=flip_vert,
                              max_rotate=max_rotate, min_zoom=min_zoom, max_zoom=max_zoom, max_warp=max_warp,
                              max_lighting=max_lighting, p_affine=p_affine, p_lighting=p_lighting, mult=mult,
                              pad_mode=pad_mode)]


class ScalePoints:
    """Points in pixels -> -1..+1 (fastai and PyTorch's grid_sample convention). Added automatically."""

    def __call__(self, sample, kinds, train):
        w, h = _img_size(sample, kinds)
        return [x / torch.tensor([w, h]) * 2 - 1 if k == "point" else x for x, k in zip(sample, kinds)]


# ==================================================================================================
# DataBlock -> Datasets -> DataLoaders
# ==================================================================================================
class BlockDataset(Dataset):
    def __init__(self, dblock, xs, ys, train):
        self.dblock, self.xs, self.ys, self.train = dblock, xs, ys, train

    def __len__(self):
        return len(self.xs)

    def raw(self, i):
        """Opened and encoded, before item transforms."""
        db = self.dblock
        sample = [db.blocks[0].create(self.xs[i])]
        if self.ys is not None:
            sample.append(db.blocks[1].encode(db.blocks[1].create(self.ys[i])))
        return sample

    def __getitem__(self, i):
        sample = self.raw(i)
        kinds = self.dblock.kinds[:len(sample)]
        for tfm in self.dblock.item_tfms:
            sample = tfm(sample, kinds, self.train)
        return tuple(_to_tensor(sample, kinds))


class DataBlock:
    """Describe your data with: blocks (input, target types), get_items, get_x, get_y, splitter,
    item_tfms (CPU, per item) and batch_tfms (GPU, per batch). Then call `.dataloaders(source)`."""

    def __init__(self, blocks=(ImageBlock, CategoryBlock), get_items=None, get_x=None, get_y=None,
                 splitter=None, item_tfms=None, batch_tfms=None, n_inp=1):
        self._kwargs = dict(blocks=blocks, get_items=get_items, get_x=get_x, get_y=get_y, splitter=splitter,
                            item_tfms=item_tfms, batch_tfms=batch_tfms, n_inp=n_inp)
        self.blocks = [b() if isinstance(b, type) else b for b in blocks]
        self.kinds = [b.kind for b in self.blocks]
        self.get_items, self.get_x, self.get_y = get_items, get_x, get_y
        self.splitter = splitter or RandomSplitter(seed=None)
        tfms = item_tfms if isinstance(item_tfms, (list, tuple)) else ([item_tfms] if item_tfms else [])
        self.item_tfms = list(tfms) + ([ScalePoints()] if "point" in self.kinds else [])
        bt = batch_tfms if isinstance(batch_tfms, (list, tuple)) else ([batch_tfms] if batch_tfms else [])
        self.batch_tfms = [IntToFloatTensor()] + list(bt)
        self.n_inp = n_inp

    def new(self, **kwargs):
        """A copy of this DataBlock with some arguments changed, e.g. bears.new(item_tfms=Resize(128))."""
        return DataBlock(**{**self._kwargs, **kwargs})

    def _items(self, source):
        items = self.get_items(source) if self.get_items else source
        if hasattr(items, "iterrows"):                   # a DataFrame: one item per row
            items = [r for _, r in items.iterrows()]
        return list(items)

    def datasets(self, source, verbose=False):
        items = self._items(source)
        tr_idx, va_idx = self.splitter(items)
        gx = self.get_x or (lambda o: o)
        xs = [gx(o) for o in items]
        ys = [self.get_y(o) for o in items] if self.get_y else None
        if ys is not None and isinstance(ys[0], str) and self.kinds[1] == "multicat":
            ys = [y.split(" ") for y in ys]
        self.blocks[1].setup([ys[i] for i in tr_idx] if ys is not None else [])
        self.vocab = getattr(self.blocks[1], "vocab", None)
        sel = lambda lst, idx: [lst[i] for i in idx] if lst is not None else None  # noqa: E731
        if verbose:
            print(f"Collecting items from {source}\nFound {len(items)} items\n2 datasets of sizes {len(tr_idx)},{len(va_idx)}")
        return (BlockDataset(self, sel(xs, tr_idx), sel(ys, tr_idx), True),
                BlockDataset(self, sel(xs, va_idx), sel(ys, va_idx), False))

    def dataloaders(self, source, bs=64, shuffle=True, num_workers=None, device=None, drop_last=None):
        train_ds, valid_ds = self.datasets(source)
        nw = default_num_workers() if num_workers is None else num_workers
        device = device or get_device()
        drop_last = (len(train_ds) > bs) if drop_last is None else drop_last
        mk = lambda ds, tr: DeviceDataLoader(  # noqa: E731
            DataLoader(ds, batch_size=bs if tr else bs * 2, shuffle=tr and shuffle, drop_last=tr and drop_last,
                       num_workers=nw, pin_memory=device.type == "cuda"),
            device, self.batch_tfms, self.kinds, train=tr, n_inp=self.n_inp)
        return BlockDataLoaders(self, mk(train_ds, True), mk(valid_ds, False), device)

    def empty_dataloaders(self, device=None):
        """DataLoaders with no data, for inference after load_learner()."""
        device = device or torch.device("cpu")
        empty = DeviceDataLoader(DataLoader([], batch_size=1), device, self.batch_tfms, self.kinds, n_inp=self.n_inp)
        return BlockDataLoaders(self, empty, empty, device)

    def summary(self, source, bs=4):
        """Print every step for the first item and try to build a batch (Ch. 5: debugging a DataBlock)."""
        train_ds, valid_ds = self.datasets(source, verbose=True)
        print(f"\nBuilding one sample from the training set: {train_ds.xs[0]}")
        raw = train_ds.raw(0)
        for x, k in zip(raw, self.kinds):
            print(f"  {k:9s} after create/encode: {_describe(x)}")
        sample = raw
        for tfm in self.item_tfms:
            sample = tfm(sample, self.kinds, True)
            print(f"  after {tfm!r}: " + ", ".join(_describe(x) for x in sample))
        print(f"\nCollating {bs} items into a batch ...")
        try:
            batch = torch.utils.data.default_collate([train_ds[i] for i in range(bs)])
            print("  batch shapes:", [tuple(t.shape) for t in batch])
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            print("  (Hint: images of different sizes can't be stacked: add item_tfms=Resize(...).)")


def _describe(x):
    if isinstance(x, Image.Image):
        return f"PIL image, mode {x.mode}, size {x.size}"
    if isinstance(x, torch.Tensor):
        return f"tensor {tuple(x.shape)} {x.tolist() if x.numel() <= 4 else ''}"
    return repr(x)


class BlockDataLoaders(DataLoaders):
    def __init__(self, dblock, train, valid, device):
        b1 = dblock.blocks[1] if len(dblock.blocks) > 1 else None
        loss = b1.loss_func() if b1 is not None and hasattr(b1, "loss_func") else None
        vocab = getattr(dblock, "vocab", None)
        c = len(vocab) if vocab is not None else (2 if b1 is not None and b1.kind == "point" else 1)
        super().__init__(train, valid, vocab=vocab, c=c, loss_func=loss, device=device, n_inp=dblock.n_inp)
        self.datablock = dblock

    train_ds = property(lambda self: self.train.dataset)
    valid_ds = property(lambda self: self.valid.dataset)

    def test_dl(self, items, bs=64):
        """A DataLoader for new, unlabelled items (file names or PIL images), with valid-set transforms."""
        ds = BlockDataset(self.datablock, list(items), None, train=False)
        return DeviceDataLoader(DataLoader(ds, batch_size=bs), self.device, self.datablock.batch_tfms,
                                self.datablock.kinds[:1], train=False, n_inp=1)

    def show_batch(self, max_n=9, ncols=None, nrows=None, unique=False, fname="batch.png", b=None, preds=None, ds_idx=0):
        """Save a grid of (decoded) images from a batch with their labels (and predictions, if given)."""
        if b is None:
            dl = self[ds_idx]
            if unique:                                    # the same image, augmented max_n times
                ds = dl.dataset
                samples = [ds[0] for _ in range(max_n)]
                b = tuple(t.to(self.device) for t in torch.utils.data.default_collate(samples))
                for tfm in dl.after_batch:
                    b = tfm(b, kinds=dl.kinds, train=True)
            else:
                b = dl.one_batch()
        kinds = self.datablock.kinds
        xs = self.decode_x(b[0][:max_n].float()).clamp(0, 1).cpu()
        imgs, titles = [], []
        palette = np.random.default_rng(0).integers(0, 255, (256, 3))
        for i in range(len(xs)):
            img = Image.fromarray((xs[i].permute(1, 2, 0).squeeze(-1).numpy() * 255).astype(np.uint8)).convert("RGB")
            title = ""
            if len(b) > 1:
                y, k, blk = b[1][i].cpu(), kinds[1], self.datablock.blocks[1]
                if k == "category":
                    title = str(blk.decode(y)) + (f"\npred: {blk.decode(preds[i])}" if preds is not None else "")
                elif k == "multicat":
                    title = ";".join(blk.decode(y))
                    if preds is not None:
                        title += "\npred: " + ";".join(blk.decode(preds[i].float()))
                elif k == "point":
                    d = ImageDraw.Draw(img)
                    pts = [(y, (255, 0, 0))] + ([(preds[i].cpu().view(-1, 2), (0, 255, 0))] if preds is not None else [])
                    for p, col in pts:
                        for px, py in p:
                            px, py = (float(px) + 1) / 2 * img.width, (float(py) + 1) / 2 * img.height
                            d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=col)
                    title = "red=target" + (" green=pred" if preds is not None else "")
                elif k == "mask":
                    m = (preds[i] if preds is not None else y).cpu().numpy()
                    overlay = Image.fromarray(palette[m].astype(np.uint8))
                    img = Image.blend(img, overlay, 0.5)
                    title = "prediction" if preds is not None else "target"
            imgs.append(img)
            titles.append(title)
        return show_images(imgs, titles, fname, ncols=ncols or min(len(imgs), 3 if max_n <= 9 else 4))


# ==================================================================================================
# Shortcuts the book uses in Chapter 1
# ==================================================================================================
class ImageDataLoaders:
    @staticmethod
    def from_name_func(path, fnames, label_func, valid_pct=0.2, seed=None, item_tfms=None, batch_tfms=None, bs=64, **kw):
        return DataBlock(blocks=(ImageBlock, CategoryBlock), splitter=RandomSplitter(valid_pct, seed),
                         get_y=label_func, item_tfms=item_tfms, batch_tfms=batch_tfms).dataloaders(fnames, bs=bs, **kw)

    @staticmethod
    def from_folder(path, train="train", valid="valid", item_tfms=None, batch_tfms=None, bs=64, **kw):
        return DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, get_y=parent_label,
                         splitter=GrandparentSplitter(train, valid), item_tfms=item_tfms,
                         batch_tfms=batch_tfms).dataloaders(path, bs=bs, **kw)


class SegmentationDataLoaders:
    @staticmethod
    def from_label_func(path, fnames, label_func, codes=None, valid_pct=0.2, seed=None, item_tfms=None,
                        batch_tfms=None, bs=8, **kw):
        return DataBlock(blocks=(ImageBlock, MaskBlock(codes)), splitter=RandomSplitter(valid_pct, seed),
                         get_y=label_func, item_tfms=item_tfms, batch_tfms=batch_tfms).dataloaders(fnames, bs=bs, **kw)


# ==================================================================================================
# Learners for images
# ==================================================================================================
def _add_imagenet_normalize(dls):
    if any(isinstance(t, Normalize) for t in dls.train.after_batch):
        return
    norm = Normalize.from_stats(*imagenet_stats)
    for dl in dls.loaders:
        dl.after_batch.append(norm)
    if hasattr(dls, "datablock"):                       # so exported models / test_dl normalise too
        dls.datablock.batch_tfms.append(norm)


def vision_learner(dls, arch, pretrained=True, loss_func=None, metrics=None, n_out=None, y_range=None,
                   normalize=True, concat_pool=True, lin_ftrs=None, ps=0.5, **kwargs):
    """A pretrained body + a new head (Ch. 1, 5 and 15). `arch` is a torchvision model function such as
    torchvision.models.resnet34. Same as fastai's vision_learner (called cnn_learner in the book)."""
    from .models import create_cnn_model, cnn_splitter
    n_out = n_out or dls.c
    model = create_cnn_model(arch, n_out, pretrained=pretrained, concat_pool=concat_pool, lin_ftrs=lin_ftrs,
                             ps=ps, y_range=y_range)
    if pretrained and normalize:
        _add_imagenet_normalize(dls)                    # pretrained weights expect ImageNet statistics
    learn = Learner(dls, model, loss_func=loss_func, metrics=metrics, splitter=cnn_splitter, **kwargs)
    if pretrained:
        learn.freeze()
    return learn


cnn_learner = vision_learner


def unet_learner(dls, arch, pretrained=True, loss_func=None, metrics=None, n_out=None, **kwargs):
    """A U-Net whose encoder is a (pretrained) ResNet (Ch. 1 and 15)."""
    from .models import SimpleUnet, cnn_splitter
    n_out = n_out or len(dls.vocab)
    model = SimpleUnet(arch, n_out, pretrained=pretrained)
    if pretrained:
        _add_imagenet_normalize(dls)
    learn = Learner(dls, model, loss_func=loss_func, metrics=metrics, splitter=cnn_splitter, **kwargs)
    if pretrained:
        learn.freeze()
    return learn
