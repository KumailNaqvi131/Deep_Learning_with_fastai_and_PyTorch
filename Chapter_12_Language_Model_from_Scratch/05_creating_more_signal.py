"""
Chapter 12 — A Language Model from Scratch
Topic 5: Creating more signal (book section of the same name)

LMModel3 predicts one word for every 3 words it reads: little feedback. Predict the next word
after EVERY word instead: use sequences of sl=16 tokens as input and the same sequences shifted
by one as targets, and output a prediction at each step. The loss must flatten the
[bs, sl, vocab] predictions and [bs, sl] targets:

    def loss_func(inp, targ):
        return F.cross_entropy(inp.view(-1, len(vocab)), targ.view(-1))

Run:
    python 05_creating_more_signal.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from common import Learner, accuracy, set_seed  # noqa: E402
from common.text import ModelResetter  # noqa: E402
from human_numbers import sequence_dls, vocab  # noqa: E402

set_seed(42)
sl = 16
dls = sequence_dls(sl)
x, y = dls.one_batch()
print("input :", " ".join(vocab[i] for i in x[0][:8]), "...")
print("target:", " ".join(vocab[i] for i in y[0][:8]), "...  (shifted by one)")


class LMModel4(nn.Module):
    def __init__(self, vocab_sz, n_hidden):
        super().__init__()
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.h_h = nn.Linear(n_hidden, n_hidden)
        self.h_o = nn.Linear(n_hidden, vocab_sz)
        self.h = 0

    def forward(self, x):
        outs = []
        for i in range(sl):
            self.h = self.h + self.i_h(x[:, i])
            self.h = F.relu(self.h_h(self.h))
            outs.append(self.h_o(self.h))
        self.h = self.h.detach()
        return torch.stack(outs, dim=1)

    def reset(self):
        self.h = 0


def loss_func(inp, targ):
    return F.cross_entropy(inp.view(-1, len(vocab)), targ.view(-1))


learn = Learner(dls, LMModel4(len(vocab), 64), loss_func=loss_func, metrics=accuracy, cbs=[ModelResetter()])
learn.fit_one_cycle(15, 3e-3)
