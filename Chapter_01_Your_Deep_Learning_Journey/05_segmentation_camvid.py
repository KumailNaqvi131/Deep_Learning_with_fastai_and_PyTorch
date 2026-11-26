"""
Chapter 1 — Your Deep Learning Journey
Topic 5: Beyond image classification: segmentation (book section "Deep Learning Is Not Just
for Image Classification")

Segmentation = classify *every pixel*: is it road, car, pedestrian, sky, ...? The book trains a
U-Net (Chapter 15 explains the architecture) on CamVid, street scenes from a car's dashboard:

    path = untar_data(URLs.CAMVID_TINY)
    dls = SegmentationDataLoaders.from_label_func(
        path, bs=8, fnames=get_image_files(path/"images"),
        label_func=lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
        codes=np.loadtxt(path/'codes.txt', dtype=str))
    learn = unet_learner(dls, resnet34)
    learn.fine_tune(8)

Each label is a PNG "mask" image with one class code (0..31) per pixel; codes.txt names them.

Run:
    python 05_segmentation_camvid.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import *  # noqa: F401,F403,E402

OUT = outputs_dir(__file__)
set_seed(42)

path = untar_data(URLs.CAMVID_TINY)
codes = np.loadtxt(path / "codes.txt", dtype=str)
print(f"{len(codes)} classes:", " ".join(codes[:12]), "...")


def label_func(o):                                   # images/scene_0001.png -> labels/scene_0001_P.png
    return path / "labels" / f"{o.stem}_P{o.suffix}"


dls = SegmentationDataLoaders.from_label_func(path, bs=8, fnames=get_image_files(path / "images"),
                                              label_func=label_func, codes=codes, seed=42)
dls.show_batch(max_n=4, fname=OUT / "camvid_batch.png")
xb, yb = dls.one_batch()
print("input batch:", tuple(xb.shape), " mask batch:", tuple(yb.shape), " (one class code per pixel)")

learn = unet_learner(dls, resnet34, metrics=foreground_acc)
learn.fine_tune(8)
learn.show_results(max_n=6, fname=OUT / "camvid_results.png")
print("Compare outputs/camvid_batch.png (targets) with outputs/camvid_results.png (predictions).")
