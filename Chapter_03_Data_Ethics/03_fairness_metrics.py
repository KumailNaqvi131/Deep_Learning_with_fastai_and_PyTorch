"""
Chapter 3 — Data Ethics
Topic 3: Measuring fairness, and historical bias (book sections "Bias", "Fairness, Accountability,
and Transparency")

Historical bias: the labels themselves reflect past discrimination. The book cites a study of
arrest data and of hiring decisions. A model trained on such labels learns to reproduce them,
however accurate it is on held-out data (which carries the same bias).

This script simulates hiring data: skill is distributed the same way in groups A and B, but past
decisions hired group B less often at equal skill. A model trained on those decisions:
  * has good accuracy against the (biased) historical labels,
  * selects group B less often (demographic parity gap),
  * accepts fewer *qualified* B candidates (equal opportunity gap).
Removing the group column does not fix it when another column (here a postcode) is a proxy for it.

There is no single "fair" metric: different definitions can conflict. The point is to measure,
discuss the trade-offs, and decide deliberately.

Run:
    python 03_fairness_metrics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from common import set_seed  # noqa: E402

set_seed(0)
n = 20000
group = (torch.rand(n) < 0.5).float()                  # 0 = A, 1 = B
skill = torch.randn(n)                                  # same distribution in both groups
qualified = (skill > 0.3).float()                       # the truth we care about
postcode = group + 0.3 * torch.randn(n)                 # correlated with group (a proxy)
experience = skill + 0.5 * torch.randn(n)
past_hired = ((skill - 0.8 * group + 0.3 * torch.randn(n)) > 0.3).float()   # biased historical decisions


def fit(features):
    x = torch.stack(features, 1)
    model = nn.Sequential(nn.Linear(x.shape[1], 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(model.parameters(), 1e-2)
    for _ in range(300):
        loss = nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(1), past_hired)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return (model(x).squeeze(1) > 0).float().detach()


def report(name, pred):
    a, b = group == 0, group == 1
    acc = (pred == past_hired).float().mean()
    sel_a, sel_b = pred[a].mean(), pred[b].mean()
    tpr_a, tpr_b = pred[a & (qualified == 1)].mean(), pred[b & (qualified == 1)].mean()
    print(f"{name}\n   accuracy vs historical labels {acc:.3f} | selection rate A {sel_a:.2f}, B {sel_b:.2f} "
          f"| qualified candidates hired: A {tpr_a:.2f}, B {tpr_b:.2f}")


report("Model using experience + group:", fit([experience, group]))
report("Model using experience + postcode (group removed):", fit([experience, postcode]))
print("Removing the protected attribute didn't help: the postcode carries the same information.\n"
      "The fix must come from the data and the process (better labels, audits, recourse), not just the code.")
