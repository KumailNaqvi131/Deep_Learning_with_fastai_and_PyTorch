"""
Chapter 15 — Application Architectures Deep Dive
Topic 2: unet_learner (book section of the same name)

For segmentation, the output must be an image as big as the input. A U-Net:
  * encoder (the "down" path): a pretrained CNN body, shrinking the image to a small grid of
    rich features (224 -> 112 -> 56 -> 28 -> 14 -> 7);
  * decoder (the "up" path): blocks that double the grid size (with nearest-neighbour upsampling
    + convs, or transposed convolutions) back to the input size;
  * skip connections: each decoder block also gets the encoder's activations at the same
    resolution, so fine details (exact edges) aren't lost.
fastai's DynamicUnet builds the decoder automatically for any encoder; common/models.py has a
simplified version (SimpleUnet).

Run:
    python 02_unet_learner.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from common import *  # noqa: F401,F403,E402
from common.models import SimpleUnet  # noqa: E402

OUT = outputs_dir(__file__)

# Upsampling: a transposed convolution vs nearest-neighbour + conv
x = torch.randn(1, 8, 7, 7)
print("ConvTranspose2d(8, 4, 2, stride=2):", tuple(nn.ConvTranspose2d(8, 4, 2, stride=2)(x).shape))
print("Upsample(scale 2) + Conv2d:        ", tuple(nn.Conv2d(8, 4, 3, padding=1)(nn.Upsample(scale_factor=2)(x)).shape))

unet = SimpleUnet(resnet34, n_out=32, pretrained=True)
feats = unet._features(torch.randn(1, 3, 96, 128))
print("\nencoder activations kept for the skip connections:", [tuple(f.shape[1:]) for f in feats])
print("U-Net output for a 96x128 image:", tuple(unet(torch.randn(1, 3, 96, 128)).shape), "= one score per class per pixel")

set_seed(42)
path = untar_data(URLs.CAMVID_TINY)
codes = np.loadtxt(path / "codes.txt", dtype=str)


def label_func(o):
    return path / "labels" / f"{o.stem}_P{o.suffix}"


dls = SegmentationDataLoaders.from_label_func(path, bs=8, fnames=get_image_files(path / "images"),
                                              label_func=label_func, codes=codes, seed=42)
learn = unet_learner(dls, resnet34, metrics=foreground_acc)
learn.fine_tune(4)
learn.show_results(max_n=4, fname=OUT / "unet_results.png")
