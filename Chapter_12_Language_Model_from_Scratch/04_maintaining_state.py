"""
Chapter 12 — A Language Model from Scratch
Topic 4: Improving the RNN: maintaining the state of an RNN (book section of the same name)

LMModel2 starts from h = 0 for every 3-word sample, throwing away everything it knew. Instead,
keep the hidden state from one batch to the next (self.h). Two consequences:
  * backpropagating through ALL previous batches would get ever slower and use ever more memory,
    so after each batch we detach() the state: gradients flow back only 3 steps (truncated
    backpropagation through time);
  * the samples must be ordered so that row i of each batch continues row i of the previous one
    (group_chunks), and the state is reset at the start of each epoch (ModelResetter callback).

    class LMModel3(Module):
        def __init__(self, vocab_sz, n_hidden):
            ...
            self.h = 0
        def forward(self, x):
            for i in range(3):
                self.h = self.h + self.i_h(x[:,i])
                self.h = F.relu(self.h_h(self.h))
            out = self.h_o(self.h)
            self.h = self.h.detach()
            return out
        def reset(self): self.h = 0

Run:
    python 04_maintaining_state.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from common import Learner, accuracy, set_seed  # noqa: E402
from common.text import ModelResetter  # noqa: E402
from human_numbers import bs, grouped_dls, three_word_dls, vocab  # noqa: E402

set_seed(42)
_, seqs = three_word_dls()
m = len(seqs) // bs
print(f"{len(seqs)} samples, bs={bs} -> m={m}: batch k holds samples k, k+m, k+2m, ... (group_chunks)")
dls = grouped_dls(seqs)


class LMModel3(nn.Module):
    def __init__(self, vocab_sz, n_hidden):
        super().__init__()
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.h_h = nn.Linear(n_hidden, n_hidden)
        self.h_o = nn.Linear(n_hidden, vocab_sz)
        self.h = 0

    def forward(self, x):
        for i in range(3):
            self.h = self.h + self.i_h(x[:, i])
            self.h = F.relu(self.h_h(self.h))
        out = self.h_o(self.h)
        self.h = self.h.detach()
        return out

    def reset(self):
        self.h = 0


learn = Learner(dls, LMModel3(len(vocab), 64), loss_func=F.cross_entropy, metrics=accuracy, cbs=[ModelResetter()])
learn.fit_one_cycle(10, 3e-3)
