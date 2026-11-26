"""
Chapter 19 — A fastai Learner from Scratch
Topic 3: Loss (book section "Loss")

Cross-entropy from scratch:
  * nll: minus the mean of the log-probabilities of the correct classes (indexing!)
  * log_softmax: log(exp(x) / sum(exp(x))) = x - log(sum(exp(x)))
  * the LogSumExp trick: exp of big numbers overflows; subtract the max first:
        log(sum(exp(x))) = m + log(sum(exp(x - m)))

Run:
    python 03_loss.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


def log_softmax_naive(x):
    return (x.exp() / (x.exp().sum(-1, keepdim=True))).log()


def logsumexp(x):
    m = x.max(-1)[0]
    return m + (x - m[:, None]).exp().sum(-1).log()


def log_softmax(x):
    return x - x.logsumexp(-1, keepdim=True)


def cross_entropy(preds, yb):
    return nll(log_softmax(preds), yb).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    r = torch.randn(64, 10)
    yb = torch.randint(0, 10, (64,))
    sm = log_softmax_naive(r)
    print("log_softmax (naive) is correct:", torch.allclose(sm, F.log_softmax(r, -1), atol=1e-6))
    print("our logsumexp == torch.logsumexp:", torch.allclose(logsumexp(r), r.logsumexp(-1), atol=1e-6))
    big = torch.tensor([[1000.0, 1001.0, 1002.0]])
    print("naive log_softmax on big activations:", log_softmax_naive(big).tolist(), "<- overflow")
    print("with the LogSumExp trick:          ", log_softmax(big).tolist())
    print(f"our cross_entropy {cross_entropy(r, yb):.6f} == F.cross_entropy {F.cross_entropy(r, yb):.6f}")
