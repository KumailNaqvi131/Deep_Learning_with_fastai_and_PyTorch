"""
Chapter 12 — A Language Model from Scratch
Topic 3: Our first recurrent neural network (book section of the same name)

LMModel1 repeats the same lines three times: write it as a for loop instead. A neural network
defined with a loop like this is a recurrent neural network (RNN). Same model, same results:

    def forward(self, x):
        h = 0
        for i in range(3):
            h = h + self.i_h(x[:,i])
            h = F.relu(self.h_h(h))
        return self.h_o(h)

Run:
    python 03_first_rnn.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from common import Learner, accuracy, set_seed  # noqa: E402
from human_numbers import three_word_dls, vocab  # noqa: E402

set_seed(42)
dls, _ = three_word_dls()


class LMModel2(nn.Module):
    def __init__(self, vocab_sz, n_hidden):
        super().__init__()
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.h_h = nn.Linear(n_hidden, n_hidden)
        self.h_o = nn.Linear(n_hidden, vocab_sz)

    def forward(self, x):
        h = 0
        for i in range(3):
            h = h + self.i_h(x[:, i])
            h = F.relu(self.h_h(h))
        return self.h_o(h)


learn = Learner(dls, LMModel2(len(vocab), 64), loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(4, 1e-3)
