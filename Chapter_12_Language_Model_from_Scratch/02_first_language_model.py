"""
Chapter 12 — A Language Model from Scratch
Topic 2: Our first language model from scratch; our language model in PyTorch (book sections of
the same names)

Predict each word from the 3 previous ones. The model uses the SAME weights for each of the 3
input words (each word is embedded, then added to the hidden state, then passed through the
same linear layer):

    class LMModel1(Module):
        def __init__(self, vocab_sz, n_hidden):
            self.i_h = nn.Embedding(vocab_sz, n_hidden)   # input -> hidden
            self.h_h = nn.Linear(n_hidden, n_hidden)      # hidden -> hidden
            self.h_o = nn.Linear(n_hidden,vocab_sz)       # hidden -> output
        def forward(self, x):
            h = F.relu(self.h_h(self.i_h(x[:,0])))
            h = h + self.i_h(x[:,1])
            h = F.relu(self.h_h(h))
            h = h + self.i_h(x[:,2])
            h = F.relu(self.h_h(h))
            return self.h_o(h)

Is ~49% accuracy good? Compare with a baseline: always predict the most common next token.

Run:
    python 02_first_language_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from common import Learner, accuracy, set_seed  # noqa: E402
from human_numbers import three_word_dls, vocab  # noqa: E402

set_seed(42)
dls, seqs = three_word_dls()
print("first samples (3 words -> next word):",
      [([vocab[i] for i in x], vocab[y]) for x, y in seqs[:3]])


class LMModel1(nn.Module):
    def __init__(self, vocab_sz, n_hidden):
        super().__init__()
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.h_h = nn.Linear(n_hidden, n_hidden)
        self.h_o = nn.Linear(n_hidden, vocab_sz)

    def forward(self, x):
        h = F.relu(self.h_h(self.i_h(x[:, 0])))
        h = h + self.i_h(x[:, 1])
        h = F.relu(self.h_h(h))
        h = h + self.i_h(x[:, 2])
        h = F.relu(self.h_h(h))
        return self.h_o(h)


learn = Learner(dls, LMModel1(len(vocab), 64), loss_func=F.cross_entropy, metrics=accuracy)
learn.fit_one_cycle(4, 1e-3)

# Baseline: the most common token in the validation set
n, counts = 0, torch.zeros(len(vocab))
for x, y in dls.valid:
    y = y.cpu()
    n += y.shape[0]
    for i in range(len(vocab)):
        counts[i] += (y == i).long().sum()
idx = torch.argmax(counts)
print(f"\nbaseline: always predict '{vocab[idx.item()]}' -> accuracy {counts[idx].item() / n:.3f}")
