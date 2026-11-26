"""
Chapter 13 — Convolutional Neural Networks
Topic 1: The magic of convolutions; mapping a convolutional kernel (book sections of the same names)

A convolution slides a small matrix, the kernel, over the image: at each position, multiply the
kernel element-wise with the 3x3 patch of pixels under it and add everything up. The kernel

    top_edge = tensor([[-1,-1,-1],
                       [ 0, 0, 0],
                       [ 1, 1, 1]]).float()

gives a large number where the pixels below are brighter than the pixels above: it detects top
edges. In a CNN the kernels are not designed by hand: they are parameters learned by SGD.

Run:
    python 01_convolution_by_hand.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from common import URLs, outputs_dir, show_images, untar_data  # noqa: E402

OUT = outputs_dir(__file__)
path = untar_data(URLs.MNIST_SAMPLE)
im3 = Image.open(sorted((path / "train" / "3").glob("*.png"))[1])
im3_t = torch.tensor(np.array(im3)).float() / 255

top_edge = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).float()
print("A 3x3 patch at the top-left:\n", im3_t[0:3, 0:3], "\n* top_edge, summed:", (im3_t[0:3, 0:3] * top_edge).sum().item())
print("a patch on the top edge of the 3 ->", round((im3_t[4:7, 6:9] * top_edge).sum().item(), 3))
print("a patch on a bottom edge         ->", round((im3_t[7:10, 17:20] * top_edge).sum().item(), 3))


def apply_kernel(row, col, kernel):
    return (im3_t[row - 1:row + 2, col - 1:col + 2] * kernel).sum()


print("apply_kernel(5, 7, top_edge) =", apply_kernel(5, 7, top_edge).item())

# Map the kernel over every position of the image (except the 1-pixel border)
rng = range(1, 27)
top_edge3 = torch.tensor([[apply_kernel(i, j, top_edge) for j in rng] for i in rng])
left_edge = torch.tensor([[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]]).float()
left_edge3 = torch.tensor([[apply_kernel(i, j, left_edge) for j in rng] for i in rng])
print("result:", tuple(top_edge3.shape), "(26x26: a 3x3 kernel can't be centred on the border pixels)")


def to_img(t):
    return ((t - t.min()) / (t.max() - t.min()))


show_images([im3_t, to_img(top_edge3), to_img(left_edge3)], ["the 3", "top_edge", "left_edge"],
            OUT / "edges_by_hand.png", ncols=3, cell=140)
print("bright = strong response: top_edge lights up the tops of the strokes, left_edge their left sides")
