"""
Chapter 6 — Other Computer Vision Problems
Topic 3: Training a multi-label model and choosing the threshold (book section "Binary Cross Entropy")

    learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.2))
    learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)
    learn.metrics = partial(accuracy_multi, thresh=0.1);  learn.validate()   # too low
    learn.metrics = partial(accuracy_multi, thresh=0.99); learn.validate()   # too high
    preds,targs = learn.get_preds()
    xs = torch.linspace(0.05,0.95,29)
    accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False) for i in xs]

Choosing the threshold on the validation set is fine here: the accuracy-vs-threshold curve is
smooth, so we're not overfitting to the validation set by picking its best point.

Run:
    python 03_train_multilabel.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functools import partial  # noqa: E402

import pandas as pd  # noqa: E402

from common import *  # noqa: F401,F403,E402

OUT = outputs_dir(__file__)
set_seed(42)
path = untar_data(URLs.PASCAL_2007)
df = pd.read_csv(path / "train.csv")
dls = DataBlock(blocks=(ImageBlock, MultiCategoryBlock), splitter=ColSplitter("is_valid"),
                get_x=ColReader("fname", pref=path / "train"), get_y=ColReader("labels", label_delim=" "),
                item_tfms=RandomResizedCrop(128, min_scale=0.35)).dataloaders(df)

learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.2))
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)

for t in (0.1, 0.99):
    learn.metrics = [partial(accuracy_multi, thresh=t)]
    print(f"validate with thresh={t}: [valid_loss, accuracy_multi] =", [round(v, 4) for v in learn.validate()])

preds, targs = learn.get_preds()                 # preds are already sigmoid-ed
print("accuracy_multi(preds, targs, thresh=0.9, sigmoid=False) =", accuracy_multi(preds, targs, thresh=0.9, sigmoid=False).item())
xs = torch.linspace(0.05, 0.95, 29)
accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False).item() for i in xs]
best = int(torch.tensor(accs).argmax())
print(f"best threshold on the validation set: {xs[best]:.2f} (accuracy {accs[best]:.4f})")

plt = try_import_matplotlib()
if plt is not None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(xs, accs)
    ax.set_xlabel("threshold")
    ax.set_ylabel("accuracy_multi")
    fig.savefig(OUT / "threshold.png", bbox_inches="tight")
    print(f"saved {OUT / 'threshold.png'}")

learn.loss_func.thresh = float(xs[best])
learn.show_results(max_n=9, fname=OUT / "multilabel_results.png")
