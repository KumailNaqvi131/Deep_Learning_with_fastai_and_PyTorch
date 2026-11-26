"""Shared by the Chapter 4 scripts: load MNIST_SAMPLE (3s and 7s) as tensors, as the book does.

    train_x: [n, 784] floats in 0..1     train_y: [n, 1] (1 for a 3, 0 for a 7)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from common import URLs, untar_data  # noqa: E402

path = untar_data(URLs.MNIST_SAMPLE)


def load_digits(folder):
    """Every image in `folder` -> one rank-3 tensor [n, 28, 28] of floats between 0 and 1."""
    files = sorted(folder.glob("*.png"))
    return torch.stack([torch.tensor(np.array(Image.open(o))) for o in files]).float() / 255


def load_xy(split):
    threes, sevens = load_digits(path / split / "3"), load_digits(path / split / "7")
    x = torch.cat([threes, sevens]).view(-1, 28 * 28)
    y = torch.tensor([1] * len(threes) + [0] * len(sevens)).unsqueeze(1)
    return x, y
