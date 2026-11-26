"""
Chapter 1 — Your Deep Learning Journey
Topic 1: Checking your setup (book section "The Software: PyTorch, fastai, and Jupyter")

The book runs everything in Jupyter notebooks with the fastai library. This repository uses
plain Python scripts and a small, readable re-implementation of the fastai pieces the book
uses (in ../common/), so the only hard requirements are PyTorch, torchvision, NumPy, pandas
and Pillow. This script checks that they are installed and tells you whether you have a GPU.

Run:
    python 01_check_setup.py
"""

import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print(f"Python      {platform.python_version()} on {platform.system()}")

required = ["torch", "torchvision", "numpy", "pandas", "PIL"]
optional = {"matplotlib": "plots (loss curves, lr_find, confusion matrices)",
            "sklearn": "decision trees and random forests (Chapter 9)",
            "scipy": "clustering similar columns (Chapter 9)",
            "sentencepiece": "subword tokenization (Chapter 10)"}

missing = []
for name in required:
    try:
        mod = importlib.import_module(name)
        print(f"{name:12s}{getattr(mod, '__version__', '')}")
    except ImportError:
        missing.append(name)
        print(f"{name:12s}MISSING (required)")
for name, why in optional.items():
    try:
        mod = importlib.import_module(name)
        print(f"{name:12s}{getattr(mod, '__version__', '')}  (optional: {why})")
    except ImportError:
        print(f"{name:12s}not installed (optional: {why})")
if missing:
    sys.exit("\nInstall the missing packages first:  pip install -r requirements.txt")

import torch  # noqa: E402

from common import get_device  # noqa: E402

device = get_device()
print(f"\nDevice used by the scripts: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB memory")
else:
    print("No GPU found: everything still runs on the CPU, just more slowly.\n"
          "(The book recommends a GPU server; Colab or Kaggle notebooks give you one for free.)")

# A tiny computation, to be sure PyTorch works on that device
x = torch.randn(1000, 1000, device=device)
print("Matrix multiply OK:", tuple((x @ x).shape))

# The datasets are generated on first use (offline). Try the smallest one.
from common import URLs, untar_data  # noqa: E402

path = untar_data(URLs.HUMAN_NUMBERS)
print("Example dataset folder:", path, "->", sorted(p.name for p in path.iterdir()))
print("\nAll good! Next: python 02_first_model_cats_vs_dogs.py")
