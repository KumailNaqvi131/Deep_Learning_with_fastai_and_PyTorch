"""
Chapter 4 — Under the Hood: Training a Digit Classifier
Topic 1: Pixels: the foundations of computer vision (book section of the same name)

To a computer an image is just numbers. MNIST_SAMPLE contains hand-written 3s and 7s, 28x28
pixels, grey-scale: each pixel is a number from 0 (black) to 255 (white).

    path = untar_data(URLs.MNIST_SAMPLE)
    threes = (path/'train'/'3').ls().sorted()
    im3 = Image.open(threes[1])
    tensor(im3)[4:10,4:10]

Run:
    python 01_pixels.py          # add --real to use the real MNIST sample (3 MB download)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from common import URLs, outputs_dir, untar_data  # noqa: E402

OUT = outputs_dir(__file__)
path = untar_data(URLs.MNIST_SAMPLE)
print("Folders:", sorted(p.name for p in path.iterdir()))
threes = sorted((path / "train" / "3").glob("*.png"))
sevens = sorted((path / "train" / "7").glob("*.png"))
print(f"{len(threes)} threes and {len(sevens)} sevens in the training set")

im3 = Image.open(threes[1])
im3.resize((140, 140), Image.NEAREST).save(OUT / "a_three.png")
print("\nThe image as a NumPy array (rows 4-9, columns 4-9):")
print(np.array(im3)[4:10, 4:10])
print("\nThe same as a PyTorch tensor:")
print(torch.tensor(np.array(im3))[4:10, 4:10])

# The book shows the pixels in a DataFrame coloured by value; in a terminal, characters do the job
t = torch.tensor(np.array(im3))
df = pd.DataFrame(t[4:24, 4:24].numpy())
print("\nRows 4-23 and columns 4-23 as a table (0 = black, 255 = white):")
print(df.to_string(header=False, index=False, col_space=3))
print("\nAs 'ASCII art' (# = bright pixel):")
for row in t:
    print("".join("#" if v > 128 else ("+" if v > 40 else ".") for v in row.tolist()))
