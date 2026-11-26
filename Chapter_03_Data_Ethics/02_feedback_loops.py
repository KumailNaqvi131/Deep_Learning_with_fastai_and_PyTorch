"""
Chapter 3 — Data Ethics
Topic 2: Feedback loops (book sections "Feedback Loops: YouTube's Recommendation System" and
"Feedback Loops")

A feedback loop happens when a model controls the data it will be trained on next. The book's
examples: recommendation systems that push ever more extreme content, and predictive policing:
police are sent where the model predicts crime; more police there means more *recorded* crime
there; the next model, trained on the recorded crime, sends even more police there.

This script simulates the predictive policing loop with two districts that have the SAME true
crime rate. Every day patrols are allocated in proportion to the crime *recorded* so far, and
only crime that happens where a patrol is can be recorded. A small random difference at the
start grows into a huge, self-confirming difference. Then the same simulation with a fix: the
model corrects for how the data was collected, using recorded crime *per patrol*, and a few
patrols always go to each district so both keep being observed.

Run:
    python 02_feedback_loops.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from common import outputs_dir, try_import_matplotlib  # noqa: E402

rng = np.random.default_rng(3)
TRUE_RATE = np.array([10.0, 10.0])           # crimes per day in districts 0 and 1: identical!
N_PATROLS = 10


def simulate(days=365, corrected=False):
    recorded = np.array([6.0, 5.0])           # a tiny difference in the historical data
    patrol_days = np.array([1.0, 1.0])
    history = []
    for _ in range(days):
        score = recorded / patrol_days if corrected else recorded   # crime per patrol vs raw counts
        share = score ** 2 / (score ** 2).sum()             # the "model": send patrols where it predicts crime
        if corrected:
            share = 0.8 * share + 0.2 * 0.5                 # ... and always keep observing both districts
        patrols = rng.multinomial(N_PATROLS, share)
        patrol_days += patrols
        crimes = rng.poisson(TRUE_RATE)
        detect = 1 - (1 - 0.06) ** patrols                  # more patrols -> more of the crime gets recorded
        recorded += rng.binomial(crimes, detect)
        history.append(recorded[0] / recorded.sum())
    return np.array(history)


h = simulate()
h_fix = simulate(corrected=True)
print("True crime rates are equal: district 0 'should' have 50% of the recorded crime.")
for day in (1, 30, 100, 365):
    print(f"  day {day:3d}: district 0 has {h[day - 1]:.0%} of recorded crime   (corrected model: {h_fix[day - 1]:.0%})")
print("\nThe model's predictions became true because the model shaped the data.\n"
      "Remedies from the book: keep humans in the loop, monitor what the model changes, collect data\n"
      "independently of the model, and think about feedback loops before deploying.")

plt = try_import_matplotlib()
if plt is not None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h, label="patrols follow the model")
    ax.plot(h_fix, label="model uses crime per patrol")
    ax.axhline(0.5, color="k", ls="--", label="truth (equal crime)")
    ax.set_xlabel("day")
    ax.set_ylabel("share of recorded crime in district 0")
    ax.set_ylim(0, 1)
    ax.legend()
    out = outputs_dir(__file__) / "feedback_loop.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"saved {out}")
