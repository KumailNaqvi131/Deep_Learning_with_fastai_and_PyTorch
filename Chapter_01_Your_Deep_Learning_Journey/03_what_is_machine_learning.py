"""
Chapter 1 — Your Deep Learning Journey
Topic 3: What is machine learning? What is a neural network? Jargon
(book sections "What Is Machine Learning?", "What Is a Neural Network?", "A Bit of Deep Learning
Jargon", "Limitations Inherent to Machine Learning")

Arthur Samuel (1962): "Suppose we arrange for some automatic means of testing the effectiveness of
any current weight assignment in terms of actual performance and provide a mechanism for altering
the weight assignment so as to maximize the performance."

This script builds Samuel's loop by hand on a tiny problem, labelling every part with the modern
jargon from the book:

    inputs ──► [ architecture + parameters ] ──► predictions ──► loss (vs labels)
                          ▲                                          │
                          └──────────── update (SGD) ◄───────────────┘

Run:
    python 03_what_is_machine_learning.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import outputs_dir, set_seed, try_import_matplotlib  # noqa: E402

set_seed(0)
OUT = outputs_dir(__file__)

# ---- A "normal" program: a human writes the rules ---------------------------------------------
def program(height, weight):
    return "adult" if height > 150 else "child"           # rule invented by a programmer


# ---- Machine learning: the rules are *learned* from examples -------------------------------
# Independent variables (inputs): height (cm) and weight (kg). Dependent variable (labels): adult?
n = 400
height = torch.cat([torch.normal(130, 15, (n // 2,)), torch.normal(172, 10, (n // 2,))])
weight = torch.cat([torch.normal(30, 8, (n // 2,)), torch.normal(72, 12, (n // 2,))])
labels = torch.cat([torch.zeros(n // 2), torch.ones(n // 2)])
inputs = torch.stack([(height - 150) / 25, (weight - 50) / 20], 1)       # scaled to roughly -2..2

# Hold some data out: the VALIDATION SET measures how well the model works on data it never saw
perm = torch.randperm(n)
train_idx, valid_idx = perm[:300], perm[300:]


# ARCHITECTURE: the functional form of the model. Here: a weighted sum, squashed by a sigmoid
def architecture(inputs, parameters):
    w, b = parameters[:2], parameters[2]
    return torch.sigmoid(inputs @ w + b)


# PARAMETERS (Samuel's "weights"): start random
parameters = torch.randn(3, requires_grad=True)


# LOSS: a measure of performance the computer can use to update the parameters (lower = better)
def loss_func(predictions, labels):
    return torch.where(labels == 1, 1 - predictions, predictions).mean()


# METRIC: a measure of performance for *people* (here accuracy)
def accuracy(predictions, labels):
    return ((predictions > 0.5).float() == labels).float().mean()


print("epoch  train_loss  valid_accuracy")
for epoch in range(30):                                    # one EPOCH = one pass through the training data
    predictions = architecture(inputs[train_idx], parameters)
    loss = loss_func(predictions, labels[train_idx])
    loss.backward()                                        # how should each parameter change to lower the loss?
    with torch.no_grad():
        parameters -= 1.0 * parameters.grad                # UPDATE: a step of gradient descent (lr = 1.0)
        parameters.grad.zero_()
        valid_acc = accuracy(architecture(inputs[valid_idx], parameters), labels[valid_idx])
    if epoch % 5 == 0 or epoch == 29:
        print(f"{epoch:5d}  {loss.item():10.4f}  {valid_acc.item():14.3f}")

print("\nLearned parameters:", parameters.data.tolist())
print("The trained MODEL = architecture + these parameters. Use it like a normal program:")
for h, w in [(120, 25), (180, 80), (155, 50)]:
    x = torch.tensor([[(h - 150) / 25, (w - 50) / 20]])
    p = architecture(x, parameters).item()
    print(f"  height {h} cm, weight {w} kg -> P(adult) = {p:.2f}   (hand-written rule says: {program(h, w)})")

# ---- Limitations (book section "Limitations Inherent to Machine Learning") -------------------
print("""
Limitations:
  * A model can only be created from data, and only learns the patterns in its training data.
  * It makes *predictions*, not recommended actions.
  * It needs *labelled* examples: someone had to say who is an adult.
  * Feedback loops: a model's predictions can change the data it is later trained on (Ch. 3).
""")

plt = try_import_matplotlib()
if plt is not None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(height[labels == 0], weight[labels == 0], s=8, label="child")
    ax.scatter(height[labels == 1], weight[labels == 1], s=8, label="adult")
    w0, w1, b = parameters.data.tolist()
    hs = torch.linspace(90, 200, 50)
    ws = 50 - 20 * (w0 * (hs - 150) / 25 + b) / w1          # where w.x + b = 0 (P = 0.5)
    ax.plot(hs, ws, "k--", label="learned boundary")
    ax.set_xlabel("height (cm)")
    ax.set_ylabel("weight (kg)")
    ax.set_ylim(0, 110)
    ax.legend()
    fig.savefig(OUT / "learned_boundary.png", bbox_inches="tight")
    print(f"saved {OUT / 'learned_boundary.png'}")
