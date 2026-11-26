"""
Chapter 9 — Tabular Modeling Deep Dive
Topic 1: Categorical embeddings (book section "Categorical Embeddings")

Continuous variables (age, price) can go straight into a model; categorical variables (day of week,
store, product id) can't: their codes have no meaningful order. The answer from collaborative
filtering: an embedding per category. An embedding layer is exactly a linear layer applied to
one-hot encoded inputs, but much faster. Guo & Berkhahn (2016, "Entity Embeddings of Categorical
Variables") showed the learned embeddings capture real structure: German states learned from
sales data ended up placed like a map of Germany.

Here: synthetic shop sales that depend on the day of week (busy weekends, a market on Wednesday)
and on the month (December rush, quiet summer). A tiny model with an embedding per categorical
column learns them. We then look at the embeddings: days with similar sales get similar vectors.

Run:
    python 01_categorical_embeddings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from common import Embedding, set_seed  # noqa: E402

set_seed(0)
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
day_effect = torch.tensor([0.0, 0.1, 0.8, 0.1, 0.3, 1.2, 1.1])         # Wed market, busy weekend
month_effect = torch.tensor([0.0, -0.1, 0.0, 0.1, 0.1, -0.3, -0.5, -0.4, 0.0, 0.1, 0.3, 1.0])
n = 20000
day = torch.randint(0, 7, (n,))
month = torch.randint(0, 12, (n,))
sales = day_effect[day] + month_effect[month] + 0.2 * torch.randn(n)

# An embedding is a linear layer on one-hot inputs:
emb = nn.Embedding(7, 3)
one_hot = F.one_hot(day[:5], 7).float()
print("one_hot @ emb.weight == emb(day):", torch.allclose(one_hot @ emb.weight, emb(day[:5])))


class SalesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.day_emb, self.month_emb = Embedding(7, 4), Embedding(12, 4)
        self.layers = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, day, month):
        return self.layers(torch.cat([self.day_emb(day), self.month_emb(month)], 1)).squeeze(1)


model = SalesModel()
opt = torch.optim.Adam(model.parameters(), 1e-2)
for epoch in range(300):
    loss = F.mse_loss(model(day, month), sales)
    opt.zero_grad()
    loss.backward()
    opt.step()
print(f"training MSE: {loss.item():.4f} (the noise alone has MSE 0.04)")

e = model.day_emb.weight.detach()
sim = F.cosine_similarity(e[:, None], e[None], dim=-1)
print("\nCosine similarity between the learned day-of-week embeddings:")
print("      " + " ".join(f"{d:>5}" for d in days))
for d, row in zip(days, sim):
    print(f"{d:>5} " + " ".join(f"{v:5.2f}" for v in row.tolist()))
for d in ("Sat", "Wed", "Mon"):
    i = days.index(d)
    closest = sorted(range(7), key=lambda j: -sim[i, j])[1]
    print(f"closest day to {d}: {days[closest]}")
print("Nobody told the model which days are alike: it learned it from the sales.")
