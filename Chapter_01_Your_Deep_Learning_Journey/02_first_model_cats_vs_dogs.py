"""
Chapter 1 — Your Deep Learning Journey
Topic 2: Your first model: is it a cat or a dog? (book sections "Your First Model",
"Running Your First Notebook" and "How Our Image Recognizer Works")

The book's six lines of fastai code, line by line:

    path = untar_data(URLs.PETS)/'images'                         # 1. get the data
    def is_cat(x): return x[0].isupper()                          # 2. how to label it
    dls = ImageDataLoaders.from_name_func(                        # 3. DataLoaders
        path, get_image_files(path), valid_pct=0.2, seed=42,
        label_func=is_cat, item_tfms=Resize(224))
    learn = cnn_learner(dls, resnet34, metrics=error_rate)       # 4. a pretrained CNN
    learn.fine_tune(1)                                            # 5. transfer learning

In the Oxford-IIIT Pet dataset, cat breeds are written with a capital letter
(Persian_12.jpg) and dog breeds in lowercase (beagle_7.jpg), so the label is in the file name.
Here the images are synthetic cartoons (see ../common/synth_vision.py); add --real to
download the real 800 MB dataset instead.

Run:
    python 02_first_model_cats_vs_dogs.py
    python 02_first_model_cats_vs_dogs.py --image my_cat.jpg      # classify your own picture
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import *  # noqa: F401,F403  (the book's `from fastai.vision.all import *`)

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="an image file to classify after training")
parser.add_argument("--real", action="store_true", help="use the real Oxford-IIIT Pet dataset (800 MB)")
args = parser.parse_args()
OUT = outputs_dir(__file__)
set_seed(42)

# 1. Get the data: a folder of images whose names start with the breed
path = untar_data(URLs.PETS) / "images"
files = get_image_files(path)
print(f"{len(files)} images, e.g. {files[0].name}, {files[-1].name}")


# 2. The label: True for cats (capitalised breed names)
def is_cat(x):
    return x[0].isupper()


# 3. DataLoaders: 20% of the images are kept aside as a validation set (seed=42 -> same split every run),
#    and every image is resized to 224x224 (Resize crops: random on train, centre on valid)
dls = ImageDataLoaders.from_name_func(path, files, valid_pct=0.2, seed=42,
                                      label_func=using_attr(is_cat, "name"), item_tfms=Resize(224))
print(f"training images: {len(dls.train_ds)}, validation images: {len(dls.valid_ds)}, labels: {dls.vocab}")
dls.show_batch(max_n=6, fname=OUT / "pets_batch.png")

# 4. A convolutional neural network (ResNet-34) pretrained on ImageNet, with a new head for our 2 classes.
#    error_rate = fraction of validation images classified wrongly (a *metric*, for humans)
learn = cnn_learner(dls, resnet34, metrics=error_rate)

# 5. fine_tune: 1 epoch training only the new head, then 1 epoch training the whole network
learn.fine_tune(1)

# Use the model: predict() returns (decoded label, label index, probabilities)
img = args.image or files[0]
is_cat_pred, _, probs = learn.predict(img)
print(f"\n{Path(img).name}: is this a cat? {is_cat_pred}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
learn.show_results(max_n=6, fname=OUT / "pets_results.png")
