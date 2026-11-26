"""
Chapter 2 — From Model to Production
Topic 4: Training your model, and using it to clean your data (book section of the same name)

    learn = cnn_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(4)
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    interp.plot_top_losses(5, nrows=1)
    cleaner = ImageClassifierCleaner(learn)

The images with the highest loss are those the model got wrong with confidence, or got right
without confidence. They are often *data* problems: mislabelled images, or images showing none of
our classes. Train a quick model BEFORE cleaning the data: the model helps you find the problems.

fastai's ImageClassifierCleaner is a notebook widget. Here the "cleaner" writes a CSV file of the
highest-loss images. Look at outputs/top_losses.png, edit the `action` column of
outputs/cleaner.csv (keep / delete / a new label such as teddy), then apply it:

    python 04_train_and_clean_data.py                      # train, write outputs/cleaner.csv
    python 04_train_and_clean_data.py --apply              # apply the actions in outputs/cleaner.csv
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import *  # noqa: F401,F403,E402

parser = argparse.ArgumentParser()
parser.add_argument("--apply", action="store_true", help="apply the actions written in outputs/cleaner.csv")
parser.add_argument("--real", action="store_true")
args = parser.parse_args()
OUT = outputs_dir(__file__)
path = DATA_ROOT / "real" / "bears" if args.real else untar_data(URLs.BEARS)

if args.apply:
    with open(OUT / "cleaner.csv") as f:
        for row in csv.DictReader(f):
            fn, action = Path(row["file"]), row["action"].strip()
            if action == "delete":
                fn.unlink(missing_ok=True)
                print("deleted", fn.name)
            elif action and action != "keep":
                shutil.move(str(fn), str(path / action / fn.name))
                print(f"moved {fn.name} -> {action}/")
    raise SystemExit

set_seed(42)
for f in verify_images(get_image_files(path)):
    f.unlink()
bears = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files,
                  splitter=RandomSplitter(valid_pct=0.2, seed=42), get_y=parent_label,
                  item_tfms=RandomResizedCrop(224, min_scale=0.5), batch_tfms=aug_transforms())
dls = bears.dataloaders(path)

learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

interp = ClassificationInterpretation.from_learner(learn)
interp.print_confusion_matrix()
interp.plot_confusion_matrix(fname=OUT / "confusion_matrix.png")
interp.plot_top_losses(10, fname=OUT / "top_losses.png")

# The "cleaner": highest-loss images of the training AND validation sets, with a suggested action
rows = []
for ds_idx, ds in ((0, dls.train_ds), (1, dls.valid_ds)):
    dl = dls[ds_idx].new(torch.utils.data.DataLoader(ds, batch_size=64), train=False)
    preds, targs, losses = learn.get_preds(dl=dl, with_loss=True)
    for i in losses.topk(6).indices.tolist():
        pred = dls.vocab[int(preds[i].argmax())]
        suggestion = pred if preds[i].max() > 0.9 else "delete"     # confident -> relabel, else look at it
        rows.append(dict(file=str(ds.xs[i]), label=dls.vocab[int(targs[i])], predicted=pred,
                         prob=f"{float(preds[i].max()):.2f}", loss=f"{float(losses[i]):.2f}", action=suggestion))
with open(OUT / "cleaner.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
show_images([PILImage.create(r["file"]) for r in rows],
            [f"{r['label']} -> {r['predicted']}\n{Path(r['file']).name}" for r in rows], OUT / "cleaner.png", ncols=6)
print(f"\nWrote {OUT / 'cleaner.csv'} ({len(rows)} highest-loss images, see outputs/cleaner.png).")
print("Check the suggested actions (keep / delete / a label), then run with --apply and retrain.")
