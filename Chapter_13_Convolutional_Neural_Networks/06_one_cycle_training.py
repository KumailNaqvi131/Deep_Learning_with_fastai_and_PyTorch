"""
Chapter 13 — Convolutional Neural Networks
Topic 6: 1cycle training (book section "1cycle Training")

Leslie Smith's 1cycle schedule: start with a low learning rate (warmup), increase it to a high
value, then decrease it to a very low value (annealing). Momentum does the opposite (0.95 ->
0.85 -> 0.95). The warmup avoids the instability of high learning rates on random initial
weights; the high rate trains fast and finds flatter (more general) minima; the annealing settles
into a good spot.

    learn.fit_one_cycle(epochs, lr)

color_dim shows the histogram of a layer's activations over time: with 1cycle, much less of the
"rising and crashing" seen with a constant high learning rate.

Run:
    python 06_one_cycle_training.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn.functional as F  # noqa: E402

from common import ActivationStats, Learner, accuracy, outputs_dir, set_seed, try_import_matplotlib  # noqa: E402
from mnist_full import get_dls, simple_cnn  # noqa: E402

OUT = outputs_dir(__file__)
set_seed(42)
dls = get_dls(512)
learn = Learner(dls, simple_cnn(), loss_func=F.cross_entropy, metrics=accuracy, cbs=[ActivationStats(with_hist=True)])
learn.fit_one_cycle(1, 0.06)

plt = try_import_matplotlib()
if plt is not None:
    sched = learn.param_scheduler                              # the callback fit_one_cycle used
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    axs[0].plot([h[0] for h in sched.hps["lr"]])
    axs[0].set_title("learning rate")
    axs[1].plot([h[0] for h in sched.hps["mom"]])
    axs[1].set_title("momentum")
    fig.savefig(OUT / "one_cycle_schedule.png", bbox_inches="tight")
    print(f"saved {OUT / 'one_cycle_schedule.png'}")
learn.activation_stats.plot_layer_stats(-2, fname=OUT / "stats_one_cycle.png")
learn.activation_stats.color_dim(-2, fname=OUT / "color_dim_one_cycle.png")
