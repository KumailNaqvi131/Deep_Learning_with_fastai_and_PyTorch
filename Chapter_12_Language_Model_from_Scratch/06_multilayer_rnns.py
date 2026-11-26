"""
Chapter 12 — A Language Model from Scratch
Topic 6: Multilayer RNNs; exploding or disappearing activations (book sections "Multilayer
RNNs", "The Model", "Exploding or Disappearing Activations")

Stack RNNs: the outputs of the first RNN are the inputs of the second. PyTorch's nn.RNN does
this (n_layers). It turns out to be *worse* than one layer here: a deep RNN is like multiplying
by the same matrix many times (2 layers x 16 time steps = 32 multiplications), and repeated
multiplication makes numbers explode or vanish: 1.1**100 = 13,781 and 0.9**100 = 0.0000266.
The gradients become useless. LSTMs (next script) were designed to fix this.

    class LMModel5(Module):
        def __init__(self, vocab_sz, n_hidden, n_layers):
            self.i_h = nn.Embedding(vocab_sz, n_hidden)
            self.rnn = nn.RNN(n_hidden, n_hidden, n_layers, batch_first=True)
            self.h_o = nn.Linear(n_hidden, vocab_sz)
            self.h = torch.zeros(n_layers, bs, n_hidden)
        def forward(self, x):
            res,h = self.rnn(self.i_h(x), self.h)
            self.h = h.detach()
            return self.h_o(res)

Run:
    python 06_multilayer_rnns.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from common import CrossEntropyLossFlat, Learner, accuracy, set_seed  # noqa: E402
from common.text import ModelResetter  # noqa: E402
from human_numbers import bs, sequence_dls, vocab  # noqa: E402

print("1.1**100 =", round(1.1 ** 100), "   0.9**100 =", 0.9 ** 100)
w = torch.randn(64, 64) / 8 * 1.2
x = torch.randn(64)
norms = []
for i in range(50):
    x = w @ x
    norms.append(x.norm().item())
print("norm of x after multiplying by the same random matrix 1, 10, 25, 50 times:",
      [f"{norms[i]:.3g}" for i in (0, 9, 24, 49)])


class LMModel5(nn.Module):
    def __init__(self, vocab_sz, n_hidden, n_layers):
        super().__init__()
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.rnn = nn.RNN(n_hidden, n_hidden, n_layers, batch_first=True)
        self.h_o = nn.Linear(n_hidden, vocab_sz)
        self.h = torch.zeros(n_layers, bs, n_hidden)

    def forward(self, x):
        res, h = self.rnn(self.i_h(x), self.h.to(x.device))
        self.h = h.detach()
        return self.h_o(res)

    def reset(self):
        self.h.zero_()


set_seed(42)
dls = sequence_dls(16)
print("\n2-layer RNN:")
learn = Learner(dls, LMModel5(len(vocab), 64, 2), loss_func=CrossEntropyLossFlat(), metrics=accuracy, cbs=[ModelResetter()])
learn.fit_one_cycle(15, 3e-3)
