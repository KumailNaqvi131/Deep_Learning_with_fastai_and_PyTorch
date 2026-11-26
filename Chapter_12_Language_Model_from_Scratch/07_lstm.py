"""
Chapter 12 — A Language Model from Scratch
Topic 7: LSTM: building an LSTM from scratch; training a language model using LSTMs (book
sections of the same names)

An LSTM keeps two states: the hidden state h (what to output now) and the cell state c (long-term
memory). Four small networks ("gates") decide, from the input and h:
  forget gate  f = sigmoid(...)   what to erase from c         c = c * f
  input gate   i = sigmoid(...)   what to update               c = c + i * g
  cell gate    g = tanh(...)      the candidate new values
  output gate  o = sigmoid(...)   what to output               h = o * tanh(c)
Because c is changed by additions and multiplications by numbers in (0, 1) rather than by a full
matrix multiplication at every step, gradients survive over many more steps.

Run:
    python 07_lstm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from common import CrossEntropyLossFlat, Learner, accuracy, set_seed  # noqa: E402
from common.text import ModelResetter  # noqa: E402
from human_numbers import bs, sequence_dls, vocab  # noqa: E402


class LSTMCell(nn.Module):
    """The book's first version: one linear layer per gate."""

    def __init__(self, ni, nh):
        super().__init__()
        self.forget_gate = nn.Linear(ni + nh, nh)
        self.input_gate = nn.Linear(ni + nh, nh)
        self.cell_gate = nn.Linear(ni + nh, nh)
        self.output_gate = nn.Linear(ni + nh, nh)

    def forward(self, input, state):
        h, c = state
        h = torch.cat([h, input], dim=1)
        forget = torch.sigmoid(self.forget_gate(h))
        c = c * forget
        inp = torch.sigmoid(self.input_gate(h))
        cell = torch.tanh(self.cell_gate(h))
        c = c + inp * cell
        out = torch.sigmoid(self.output_gate(h))
        h = out * torch.tanh(c)
        return h, (h, c)


class LSTMCell2(nn.Module):
    """Refactored: one big matrix multiplication for all 4 gates, then chunk the result in 4."""

    def __init__(self, ni, nh):
        super().__init__()
        self.ih = nn.Linear(ni, 4 * nh)
        self.hh = nn.Linear(nh, 4 * nh)

    def forward(self, input, state):
        h, c = state
        gates = (self.ih(input) + self.hh(h)).chunk(4, 1)
        ingate, forgetgate, outgate = map(torch.sigmoid, gates[:3])
        cellgate = gates[3].tanh()
        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * c.tanh()
        return h, (h, c)


print("t = torch.arange(0,10); t.chunk(2) =", torch.arange(0, 10).chunk(2))

# Our refactored cell computes the same thing as PyTorch's nn.LSTMCell. PyTorch stores the 4 gates in the
# order (input, forget, cell, output); our chunk order is (input, forget, output, cell), so copy accordingly.
torch.manual_seed(0)
ref = nn.LSTMCell(5, 3)
mine = LSTMCell2(5, 3)


def reorder(t):
    i, f, g, o = t.chunk(4, 0)
    return torch.cat([i, f, o, g])


with torch.no_grad():
    mine.ih.weight.copy_(reorder(ref.weight_ih))
    mine.ih.bias.copy_(reorder(ref.bias_ih))
    mine.hh.weight.copy_(reorder(ref.weight_hh))
    mine.hh.bias.copy_(reorder(ref.bias_hh))
x, h0, c0 = torch.randn(2, 5), torch.randn(2, 3), torch.randn(2, 3)
h_mine, (_, c_mine) = mine(x, (h0, c0))
h_ref, c_ref = ref(x, (h0, c0))
print("our LSTM cell == nn.LSTMCell:", torch.allclose(h_mine, h_ref, atol=1e-6) and torch.allclose(c_mine, c_ref, atol=1e-6))


class LMModel6(nn.Module):
    def __init__(self, vocab_sz, n_hidden, n_layers):
        super().__init__()
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True)
        self.h_o = nn.Linear(n_hidden, vocab_sz)
        self.h = [torch.zeros(n_layers, bs, n_hidden) for _ in range(2)]

    def forward(self, x):
        res, h = self.rnn(self.i_h(x), [t.to(x.device) for t in self.h])
        self.h = [h_.detach() for h_ in h]
        return self.h_o(res)

    def reset(self):
        for h in self.h:
            h.zero_()


set_seed(42)
dls = sequence_dls(16)
print("\n2-layer LSTM:")
learn = Learner(dls, LMModel6(len(vocab), 64, 2), loss_func=CrossEntropyLossFlat(), metrics=accuracy, cbs=[ModelResetter()])
learn.fit_one_cycle(15, 1e-2)
print("Much better than the 2-layer RNN of 06_multilayer_rnns.py, but it overfits: next, regularization.")
