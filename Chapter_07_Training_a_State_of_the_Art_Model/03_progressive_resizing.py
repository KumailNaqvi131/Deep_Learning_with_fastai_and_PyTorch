"""
Chapter 7 — Training a State-of-the-Art Model
Topic 3: Progressive resizing (book section "Progressive Resizing")

Start training with small images (fast: fewer pixels), finish with bigger ones (better accuracy).
Like transfer learning, the model trained at the small size gets fine-tuned at the larger size;
the change of size is also a kind of data augmentation, so it generalises better.

    dls = get_dls(128, 128)
    learn = Learner(dls, xresnet50(n_out=dls.c), loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    learn.fit_one_cycle(4, 3e-3)
    learn.dls = get_dls(64, 224)
    learn.fine_tune(5, 1e-3)

(We go from 64px to 128px, the equivalent for our 160px images.)

Run:
    python 03_progressive_resizing.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import CrossEntropyLossFlat, Learner, accuracy, set_seed, xresnet50  # noqa: E402
from imagenette_data import get_dls  # noqa: E402

set_seed(42)
t0 = time.time()
dls = get_dls(128, 64)
learn = Learner(dls, xresnet50(n_out=dls.c), loss_func=CrossEntropyLossFlat(), metrics=accuracy)
print("4 epochs at 64x64 pixels:")
learn.fit_one_cycle(4, 3e-3)

learn.dls = get_dls(64, 128)
print("\nthen fine_tune at 128x128 (fine_tune trains 1 epoch 'frozen' first; with a single parameter group"
      " nothing is frozen, so it's just one more epoch):")
learn.fine_tune(5, 1e-3)
print(f"total time {time.time() - t0:.0f}s")
