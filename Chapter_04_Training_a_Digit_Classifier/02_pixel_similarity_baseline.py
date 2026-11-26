"""
Chapter 4 — Under the Hood: Training a Digit Classifier
Topic 2: First try: pixel similarity (book section of the same name)

A *baseline*: a simple model you are confident works reasonably well, to compare better models
against. Idea: compute the average 3 and the average 7 ("ideal" digits). To classify an image,
see which of the two ideal digits it is most similar to.

How to measure "distance" between two images? Taking the plain difference doesn't work (positive
and negative differences cancel out), so:
  * mean absolute difference = L1 norm          (a-b).abs().mean()
  * root mean squared error  = L2 norm          ((a-b)**2).mean().sqrt()

Run:
    python 02_pixel_similarity_baseline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # for `common`  (this folder is on the path already)

import torch.nn.functional as F  # noqa: E402

from common import outputs_dir, show_images  # noqa: E402
from mnist_data import load_digits, path  # noqa: E402

OUT = outputs_dir(__file__)

# Stack all the 3s into one rank-3 tensor: [number of images, 28 rows, 28 columns]
stacked_threes = load_digits(path / "train" / "3")
stacked_sevens = load_digits(path / "train" / "7")
print("stacked_threes.shape:", tuple(stacked_threes.shape), "| rank (number of axes):", stacked_threes.ndim)

# The "ideal" 3 and 7: the mean over the image axis (axis 0)
mean3 = stacked_threes.mean(0)
mean7 = stacked_sevens.mean(0)
show_images([mean3, mean7], ["ideal 3", "ideal 7"], OUT / "ideal_digits.png", ncols=2)

a_3 = stacked_threes[1]
dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 - mean3) ** 2).mean().sqrt()
dist_7_abs = (a_3 - mean7).abs().mean()
dist_7_sqr = ((a_3 - mean7) ** 2).mean().sqrt()
print(f"\nOne 3 compared with the ideal 3: L1 {dist_3_abs:.4f}, RMSE {dist_3_sqr:.4f}")
print(f"The same 3 compared with the ideal 7: L1 {dist_7_abs:.4f}, RMSE {dist_7_sqr:.4f}")
print("-> closer to the ideal 3 with both measures: correct!")

# PyTorch already has both as loss functions
print("F.l1_loss:", F.l1_loss(a_3.float(), mean7).item(), " F.mse_loss(...).sqrt():", F.mse_loss(a_3, mean7).sqrt().item())
print("\nNext: 03_tensors_and_broadcasting.py measures the accuracy of this idea on the whole validation set.")
