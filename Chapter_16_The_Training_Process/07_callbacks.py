"""
Chapter 16 — The Training Process
Topic 7: Callbacks: creating a callback; callback ordering and exceptions (book sections "Callbacks",
"Creating a Callback", "Callback Ordering and Exceptions")

Instead of copying the training loop and modifying it for every new idea (mixup, AR/TAR, ...),
fastai's loop calls callbacks at every step, and a callback can read and CHANGE everything in
the Learner (the batch, the predictions, the loss, the model, ...). Events, in order:

    before_fit, before_epoch, before_train, before_batch, after_pred, after_loss,
    before_backward, after_backward, after_step, after_batch, after_train,
    before_validate, after_validate, after_epoch, after_fit

A callback can also interrupt training by raising an exception: CancelBatchException (skip the
rest of this batch), CancelTrainException, CancelValidException, CancelEpochException,
CancelFitException (stop everything). `order` decides which callback runs first.

Run:
    python 07_callbacks.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math  # noqa: E402

from common import Callback, CancelBatchException, CancelFitException, Learner, accuracy, set_seed  # noqa: E402
from common.learner import EVENTS  # noqa: E402
from common.text import ModelResetter  # noqa: E402
from training_setup import dls  # noqa: E402

print("events:", ", ".join(EVENTS))
print("\nfastai's ModelResetter is three lines of callback:\n"
      "    class ModelResetter(Callback):\n"
      "        def before_train(self):    self.model.reset()\n"
      "        def before_validate(self): self.model.reset()\n")
print(f"(ours: {ModelResetter.__name__} in common/text.py)\n")


class EventPrinter(Callback):
    """Print the events of the first two batches (and the end of the epoch), to see the order."""

    def before_fit(self):
        self.n = 0
        print("   event: before_fit")

    def __getattr__(self, name):
        if name in EVENTS:
            def f():
                if self.n < 18 or name in ("after_train", "before_validate", "after_validate", "after_epoch", "after_fit"):
                    print("   event:", name)
                self.n += 1
            return f
        raise AttributeError(name)


class TerminateOnNaNCallback(Callback):
    """Stop training as soon as the loss is NaN or infinite."""
    run_before = None

    def after_batch(self):
        if math.isnan(self.learn.loss.item()) or math.isinf(self.learn.loss.item()):
            print(f"   loss is {self.learn.loss.item()} at batch {self.learn.iter}: stopping")
            raise CancelFitException


class SkipEveryOtherBatch(Callback):
    """CancelBatchException: skip the rest of a batch (here: every other training batch is ignored)."""

    def before_fit(self):
        self.skipped = 0

    def after_pred(self):
        if self.learn.training and self.learn.iter % 2 == 1:
            self.skipped += 1
            raise CancelBatchException


class Timer(Callback):
    order = 100                                  # run after the others

    def before_epoch(self):
        self.t = time.time()

    def after_epoch(self):
        print(f"   (epoch {self.learn.epoch} took {time.time() - self.t:.1f}s)")


from torch import nn  # noqa: E402

from common import Flatten  # noqa: E402


def small_model():
    return nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(32, dls.c))


set_seed(42)
print("1. The order of events (EventPrinter):")
learn = Learner(dls, small_model(), loss_func=nn.CrossEntropyLoss(), metrics=accuracy, cbs=[EventPrinter()])
learn.silent = True
learn.fit(1, 1e-3)

print("\n2. A ridiculous learning rate + TerminateOnNaNCallback:")
learn = Learner(dls, small_model(), loss_func=nn.CrossEntropyLoss(), metrics=accuracy, cbs=[TerminateOnNaNCallback()])
learn.fit(1, 1e4)

print("\n3. CancelBatchException, and a Timer with order=100:")
skip = SkipEveryOtherBatch()
learn = Learner(dls, small_model(), loss_func=nn.CrossEntropyLoss(), metrics=accuracy, cbs=[skip, Timer()])
learn.fit(1, 1e-3)
print(f"   skipped {skip.skipped} of {len(dls.train)} training batches")
print("   callbacks in the order they run:", [type(c).__name__ for c in learn.cbs])
