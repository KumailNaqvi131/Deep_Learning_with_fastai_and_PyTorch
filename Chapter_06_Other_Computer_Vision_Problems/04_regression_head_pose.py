"""
Chapter 6 — Other Computer Vision Problems
Topic 4: Regression: finding the centre of a face (book sections "Regression", "Assembling the
Data", "Training a Model")

Image regression: the dependent variable is a number (or several). Here the (x, y) coordinates of
the centre of the head, in the Biwi Kinect Head Pose dataset (24 people, folders 01..24).
  * get_ctr: the head centre comes from the *_pose.txt file and the camera calibration rgb.cal;
  * PointBlock: tells fastai the target is a point, so data augmentation moves the point together
    with the image (rotate the image -> rotate the point);
  * FuncSplitter: validate on one person the model has never seen (person 13), not a random split;
  * coordinates are rescaled to -1..+1, and y_range=(-1,1) makes the model output in that range;
  * the loss is MSE (mean squared error).

    biwi = DataBlock(blocks=(ImageBlock, PointBlock), get_items=get_image_files, get_y=get_ctr,
                     splitter=FuncSplitter(lambda o: o.parent.name=='13'),
                     batch_tfms=[*aug_transforms(size=(240,320)), Normalize.from_stats(*imagenet_stats)])
    learn = cnn_learner(dls, resnet18, y_range=(-1,1))
    learn.fine_tune(3, lr)

Run:
    python 04_regression_head_pose.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math  # noqa: E402

from common import *  # noqa: F401,F403,E402

OUT = outputs_dir(__file__)
set_seed(42)
path = untar_data(URLs.BIWI_HEAD_POSE)
print("path.ls():", sorted(p.name for p in path.iterdir())[:8], "...")
img_files = get_image_files(path)


def img2pose(x):
    return Path(f"{str(x)[:-7]}pose.txt")


print("an image and its pose file:", img_files[0].relative_to(path), "->", img2pose(img_files[0]).relative_to(path))
im = PILImage.create(img_files[0])
print("image size:", im.size[::-1])

cal = np.genfromtxt(path / "01" / "rgb.cal", skip_footer=6)


def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0] / ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1] / ctr[2] + cal[1][2]
    return torch.tensor([c1, c2])


print("get_ctr(first image):", get_ctr(img_files[0]))


def is_person_13(o):
    return o.parent.name == "13"


biwi = DataBlock(blocks=(ImageBlock, PointBlock), get_items=get_image_files, get_y=get_ctr,
                 splitter=FuncSplitter(is_person_13),
                 batch_tfms=[*aug_transforms(size=(240, 320)), Normalize.from_stats(*imagenet_stats)])
dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, fname=OUT / "biwi_batch.png")
xb, yb = dls.one_batch()
print("batch shapes:", tuple(xb.shape), tuple(yb.shape), "| one target (rescaled to -1..1):", yb[0].tolist())

learn = cnn_learner(dls, resnet18, y_range=(-1, 1))
print("loss function picked automatically:", dls.loss_func)
print("sigmoid_range(tensor([-4., 0., 4.]), -1, 1) =", sigmoid_range(torch.tensor([-4.0, 0, 4]), -1, 1).tolist())

lr_min, lr_steep = learn.lr_find(fname=OUT / "biwi_lr_find.png")
lr = 1e-2
learn.fine_tune(3, lr)
mse_valid = learn.validate()[0]
print(f"\nvalid MSE {mse_valid:.5f} -> RMSE {math.sqrt(mse_valid):.4f} in -1..1 units "
      f"= about {math.sqrt(mse_valid) * 320:.1f} pixels horizontally on a 640-pixel-wide image")
learn.show_results(max_n=6, fname=OUT / "biwi_results.png")
