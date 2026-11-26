"""
Chapter 4 — Under the Hood: Training a Digit Classifier
Topic 3: NumPy arrays and PyTorch tensors; computing metrics using broadcasting (book sections
of the same names)

Tensors are like NumPy arrays (a multidimensional table of numbers of one type) that can also
live on the GPU and compute gradients. Two ideas make them fast: do operations on whole tensors
at once (no Python loops), and *broadcasting*: when shapes differ, the smaller tensor is
virtually expanded to match the larger one, without copying memory.

    def mnist_distance(a,b): return (a-b).abs().mean((-1,-2))
    mnist_distance(valid_3_tens, mean3)          # [1010,28,28] - [28,28] -> 1010 distances

Run:
    python 03_tensors_and_broadcasting.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # for `common`  (this folder is on the path already)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mnist_data import load_digits, path  # noqa: E402

# ---- NumPy arrays vs. PyTorch tensors ------------------------------------------------------------
data = [[1, 2, 3], [4, 5, 6]]
arr, tns = np.array(data), torch.tensor(data)
print("array:\n", arr, "\ntensor:\n", tns)
print("a row:", tns[1], "| a column:", tns[:, 1], "| a slice:", tns[1, 1:3])
print("tns + 1:", (tns + 1).tolist(), "| tns.type():", tns.type(), "| tns * 1.5:", (tns * 1.5).tolist())

# ---- Broadcasting -------------------------------------------------------------------------------
print("\ntensor([1,2,3]) + tensor(1) =", (torch.tensor([1, 2, 3]) + torch.tensor(1)).tolist(), "(the 1 is broadcast)")

mean3 = load_digits(path / "train" / "3").mean(0)
mean7 = load_digits(path / "train" / "7").mean(0)
valid_3_tens = load_digits(path / "valid" / "3")
valid_7_tens = load_digits(path / "valid" / "7")
print("valid_3_tens:", tuple(valid_3_tens.shape), " valid_7_tens:", tuple(valid_7_tens.shape))


def mnist_distance(a, b):
    """Mean absolute difference over the last two axes (rows and columns)."""
    return (a - b).abs().mean((-1, -2))


print("\n(valid_3_tens - mean3).shape =", tuple((valid_3_tens - mean3).shape), "<- mean3 broadcast over every image")
valid_3_dist = mnist_distance(valid_3_tens, mean3)
print("distances:", tuple(valid_3_dist.shape), valid_3_dist[:5])


def is_3(x):
    return mnist_distance(x, mean3) < mnist_distance(x, mean7)


print("\nis_3(one 3):", is_3(valid_3_tens[0]).item(), "| as a float:", is_3(valid_3_tens[0]).float().item())
accuracy_3s = is_3(valid_3_tens).float().mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()
print(f"Accuracy on 3s: {accuracy_3s:.4f}, on 7s: {accuracy_7s:.4f}, average: {(accuracy_3s + accuracy_7s) / 2:.4f}")
print("A good baseline! But it has nothing to *learn*. Next: stochastic gradient descent.")
