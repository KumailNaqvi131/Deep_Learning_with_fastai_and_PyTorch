"""
Chapter 2 — From Model to Production
Topic 5: Using the model for inference (book section "Using the Model for Inference")

A model = architecture + trained parameters. To use it in an application we also need the data
pipeline: how to open an image, resize it, normalise it, and the names of the classes. fastai's
`learn.export()` saves all of that in one file, and `load_learner` loads it back:

    learn.export()                                    # -> export.pkl
    learn_inf = load_learner(path/'export.pkl')
    learn_inf.predict('images/grizzly.jpg')
    # ('grizzly', tensor(1), tensor([9.0767e-06, 9.9999e-01, 1.5748e-07]))
    learn_inf.dls.vocab                               # ['black', 'grizzly', 'teddy']

Note: export uses Python's pickle, so any function used in the DataBlock (get_y, ...) must be
importable when you load it: define them in a module, not as lambdas.

Run:
    python 05_export_and_inference.py
    python 05_export_and_inference.py --image my_bear.jpg
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import *  # noqa: F401,F403,E402

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="an image to classify")
parser.add_argument("--real", action="store_true")
args = parser.parse_args()
OUT = outputs_dir(__file__)
path = DATA_ROOT / "real" / "bears" if args.real else untar_data(URLs.BEARS)
set_seed(42)

# ---- train (quickly) and export ------------------------------------------------------------------
for f in verify_images(get_image_files(path)):
    f.unlink()
bears = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files,
                  splitter=RandomSplitter(valid_pct=0.2, seed=42), get_y=parent_label,
                  item_tfms=RandomResizedCrop(224, min_scale=0.5), batch_tfms=aug_transforms())
learn = cnn_learner(bears.dataloaders(path), resnet18, metrics=error_rate, path=OUT)
learn.fine_tune(3)
learn.export("export.pkl")
print("files in outputs/:", sorted(p.name for p in OUT.glob("*.pkl")))

# ---- load it back (this is what your application does) ------------------------------------------
learn_inf = load_learner(OUT / "export.pkl")
img = args.image or get_image_files(path / "grizzly")[0]
pred, pred_idx, probs = learn_inf.predict(img)
print(f"\nPrediction for {Path(img).name}: {pred} (class index {int(pred_idx)})")
print("Probabilities:", {c: round(float(p), 5) for c, p in zip(learn_inf.dls.vocab, probs)})
print("learn_inf.dls.vocab =", learn_inf.dls.vocab)

# predict also accepts PIL images (e.g. from an upload)
from PIL import Image  # noqa: E402
print("From a PIL image:", learn_inf.predict(Image.open(get_image_files(path / "teddy")[0]))[0])
