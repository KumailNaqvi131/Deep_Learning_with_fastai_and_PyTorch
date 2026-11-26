"""
Chapter 5 — Image Classification
Topic 2: Presizing (book section "Presizing")

Every augmentation that rotates, zooms or warps an image has to *interpolate* new pixel values,
which loses quality, and can leave empty (reflected) zones at the borders. Doing several of them
one after another makes it worse. fastai's presizing strategy:

  1. item_tfms=Resize(460): resize every image to a fairly large size (bigger than the final 224),
     on the CPU, so all images have the same size and can be batched for the GPU;
  2. batch_tfms=aug_transforms(size=224, min_scale=0.75): compose ALL the augmentations
     (crop, rotate, zoom, warp, ...) into one transform and the final resize to 224, then do a
     single interpolation, on the GPU.

This script makes the book's comparison picture: the "usual" way (resize to 224, then rotate,
then zoom: three interpolations) vs presizing (one interpolation from 460px). Look at the edges
and fine details in outputs/presizing_comparison.png.

Run:
    python 02_presizing.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402

from common import Resize, outputs_dir, show_images  # noqa: E402
from common.vision import _to_tensor  # noqa: E402
from pets_data import get_image_files, path  # noqa: E402

OUT = outputs_dir(__file__)
img = Image.open(get_image_files(path / "images")[5]).convert("RGB")
angle, zoom = 25.0, 1.4

# The usual way: three separate interpolations at the final size
a = img.resize((224, 224), Image.BILINEAR)
a = TF.rotate(a, angle, interpolation=TF.InterpolationMode.BILINEAR)           # corners become empty
a = TF.resize(TF.center_crop(a, int(224 / zoom)), [224, 224], interpolation=TF.InterpolationMode.BILINEAR)

# Presizing: resize to 460 (item transform), then ONE affine transform (rotation + zoom + resize to 224)
big = Resize(460)([img], ["image"], train=False)[0]
x = _to_tensor([big], ["image"])[0].float().unsqueeze(0) / 255
theta = torch.zeros(1, 2, 3)
r = math.radians(angle)
theta[0, :, :2] = torch.tensor([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]]) / zoom
grid = torch.nn.functional.affine_grid(theta, (1, 3, 224, 224), align_corners=False)
b = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)[0]

show_images([img, a, b], ["original", "usual: 3 interpolations", "presizing: 1 interpolation"],
            OUT / "presizing_comparison.png", ncols=3, cell=300)
print("The usual way has empty corners and is blurrier: compare the whiskers and the fur pattern.")
