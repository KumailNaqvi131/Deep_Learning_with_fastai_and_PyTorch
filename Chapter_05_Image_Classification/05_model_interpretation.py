"""
Chapter 5 — Image Classification
Topic 5: Model interpretation (book section "Model Interpretation")

With 37 classes the confusion matrix is hard to read; most_confused lists the cells with the
most errors. The mistakes should make sense: breeds that look alike (in the real data,
Staffordshire bull terrier vs American pit bull terrier; in our synthetic data, for instance
breeds with the same fur colour and pattern).

    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    interp.most_confused(min_val=5)

Run:
    python 05_model_interpretation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ClassificationInterpretation, cnn_learner, error_rate, outputs_dir, resnet34, set_seed  # noqa: E402
from common.synth_vision import PET_BREEDS  # noqa: E402
from pets_data import get_dls  # noqa: E402

OUT = outputs_dir(__file__)
set_seed(42)
learn = cnn_learner(get_dls(), resnet34, metrics=error_rate)
learn.fine_tune(2)

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(fname=OUT / "confusion_matrix.png", figsize=(12, 12))
print("\nmost_confused(min_val=3)  (actual, predicted, count):")
for actual, pred, n in interp.most_confused(min_val=3):
    a, p = PET_BREEDS.get(actual), PET_BREEDS.get(pred)
    why = f"   both {a[0]} fur, {a[1]} / {p[1]} pattern" if a and p and a[0] == p[0] else ""
    print(f"  {actual:28s} -> {pred:28s} {n:3d}{why}")
interp.plot_top_losses(9, fname=OUT / "top_losses.png")
