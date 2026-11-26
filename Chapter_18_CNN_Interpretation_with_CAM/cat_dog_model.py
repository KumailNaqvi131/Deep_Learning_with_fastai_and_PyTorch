"""Shared by the Chapter 18 scripts: Chapter 1's cat-vs-dog model (trained once, then cached), a test
image, and a helper to draw a heat map over an image.

    dls = ImageDataLoaders.from_name_func(path, get_image_files(path), valid_pct=0.2, seed=21,
                                          label_func=is_cat, item_tfms=Resize(224))
    learn = cnn_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(1)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from common import (ImageDataLoaders, Resize, URLs, cnn_learner, error_rate, get_image_files, outputs_dir,  # noqa: E402
                    resnet34, set_seed, untar_data, using_attr)

OUT = outputs_dir(__file__)
path = untar_data(URLs.PETS) / "images"


def is_cat(x):
    return x[0].isupper()


def get_learner():
    set_seed(21)
    dls = ImageDataLoaders.from_name_func(path, get_image_files(path), valid_pct=0.2, seed=21,
                                          label_func=using_attr(is_cat, "name"), item_tfms=Resize(224))
    learn = cnn_learner(dls, resnet34, metrics=error_rate, path=OUT)
    if (OUT / "cat_dog.pth").exists():
        learn.load("cat_dog")
    else:
        print("training the cat/dog model of Chapter 1 (first time only) ...")
        learn.fine_tune(1)
        learn.save("cat_dog")
    return learn


def a_cat_image():
    """A cat picture from the validation part of the data (like the book's image_cat())."""
    return Image.open(sorted(path.glob("Bengal_*.jpg"))[-1]).convert("RGB")


def heatmap_over(img, heat, fname, title=""):
    """Draw `heat` (a small 2D grid) over the image, stretched to its size, bilinear like the book."""
    from common import try_import_matplotlib
    plt = try_import_matplotlib()
    heat = heat.detach().float().cpu()
    if plt is None:
        h = F.interpolate(heat[None, None], size=img.size[::-1], mode="bilinear")[0, 0]
        h = ((h - h.min()) / (h.max() - h.min() + 1e-8) * 255).byte().numpy()
        overlay = Image.fromarray(np.stack([h, h // 3, 255 - h], -1).astype(np.uint8))
        Image.blend(img.convert("RGB"), overlay, 0.5).save(fname)
    else:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.imshow(heat, alpha=0.6, extent=(0, img.width, img.height, 0), interpolation="bilinear", cmap="magma")
        ax.set_title(title)
        ax.axis("off")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
    print(f"saved {fname}")


def prepare(learn, img):
    """The image as the model sees it (resized, normalised), as a batch of one."""
    x, = next(iter(learn.dls.test_dl([img])))
    return x
