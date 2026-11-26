"""Small utilities shared by every chapter: folders, device, seeding, optional imports, image grids."""

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Folder layout -------------------------------------------------------------
BOOK_ROOT = Path(__file__).resolve().parents[1]    # .../Deep_Learning_for_Coders_with_fastai_and_PyTorch
DATA_ROOT = BOOK_ROOT / "data"                      # generated / downloaded datasets live here


def outputs_dir(script_file: str) -> Path:
    """Return `<chapter folder>/outputs/`, creating it if needed.  Usage: OUT = outputs_dir(__file__)"""
    out = Path(script_file).resolve().parent / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def chapter_outputs(chapter_number: int) -> Path:
    """The `outputs/` folder of another chapter (used to reuse a model trained earlier)."""
    matches = sorted(BOOK_ROOT.glob(f"Chapter_{chapter_number:02d}_*"))
    out = matches[0] / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def use_real_data() -> bool:
    """True when the script was started with `--real` (or REAL_DATA=1): download the book's dataset
    instead of generating the offline synthetic look-alike."""
    return "--real" in sys.argv or os.environ.get("REAL_DATA") == "1"


def get_device() -> torch.device:
    """Pick the best available device: NVIDIA GPU > Apple GPU > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Make runs repeatable (same random numbers every time)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def try_import_matplotlib():
    """Return matplotlib.pyplot if installed, else None (plots become optional)."""
    try:
        import matplotlib
        matplotlib.use("Agg")          # draw to files, no window needed
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[info] matplotlib not installed -> skipping plot. Install with: pip install matplotlib")
        return None


def try_import(name: str, pip_name: str = None):
    """Import an optional package, or print how to install it and return None."""
    try:
        return __import__(name)
    except ImportError:
        print(f"[info] optional package '{name}' is not installed. Install with: pip install {pip_name or name}")
        return None


def num_workers() -> int:
    """DataLoader worker processes: a few on Linux, 0 on Windows and macOS (their 'spawn' start
    method can't send lambdas to worker processes, and the data here is small anyway)."""
    return 0 if os.name == "nt" or sys.platform == "darwin" else min(4, os.cpu_count() or 1)


# Showing images without matplotlib ----------------------------------------------------------------
def to_pil(img):
    """Tensor [C,H,W] (0..1 or 0..255) / NumPy array / PIL image -> PIL image."""
    from PIL import Image
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().float()
        if img.max() <= 1.01:
            img = img * 255
        if img.ndim == 3:
            img = img.permute(1, 2, 0)
        img = img.clamp(0, 255).byte().numpy()
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    return Image.fromarray(img.astype(np.uint8))


def show_images(images, titles=None, fname="grid.png", ncols=4, cell=160):
    """Save a grid of images with captions to `fname` (uses only Pillow). Returns the path."""
    from PIL import Image, ImageDraw
    images = [to_pil(im).convert("RGB") for im in images]
    titles = titles or [""] * len(images)
    ncols = min(ncols, len(images))
    nrows = (len(images) + ncols - 1) // ncols
    cap = 28 if any(titles) else 0
    grid = Image.new("RGB", (ncols * cell, nrows * (cell + cap)), "white")
    draw = ImageDraw.Draw(grid)
    for i, (im, t) in enumerate(zip(images, titles)):
        r, c = divmod(i, ncols)
        im = im.copy()
        im.thumbnail((cell - 4, cell - 4))
        x, y = c * cell + (cell - im.width) // 2, r * (cell + cap) + cap
        grid.paste(im, (x, y))
        lines = str(t).split("\n")
        for j, line in enumerate(lines[:2]):
            draw.text((c * cell + 3, r * (cell + cap) + 2 + 12 * j), line[:26], fill="black")
    grid.save(fname)
    print(f"saved {fname}")
    return fname
