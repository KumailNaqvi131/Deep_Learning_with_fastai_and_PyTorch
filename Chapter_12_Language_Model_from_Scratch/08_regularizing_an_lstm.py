"""
Chapter 12 — A Language Model from Scratch
Topic 8: Regularizing an LSTM: dropout, AR and TAR, weight tying (book sections "Regularizing an
LSTM", "Dropout", "Activation Regularization and Temporal Activation Regularization",
"Training a Weight-Tied Regularized LSTM")

The AWD-LSTM ideas (Merity et al., 2017):
  * dropout: randomly zero activations during training (and scale the rest by 1/(1-p));
  * AR: add alpha * mean(activations**2) to the loss -> keeps activations small;
  * TAR: add beta * mean((h_t - h_{t-1})**2) -> consecutive hidden states shouldn't jump around;
  * weight tying: the output layer (hidden -> word) uses the same weights as the input embedding
    (word -> hidden): self.h_o.weight = self.i_h.weight.

    class Dropout(Module):
        def __init__(self, p): self.p = p
        def forward(self, x):
            if not self.training: return x
            mask = x.new(*x.shape).bernoulli_(1-p)
            return x * mask.div_(1-p)

Run:
    python 08_regularizing_an_lstm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from common import Callback, CrossEntropyLossFlat, Learner, accuracy, set_seed  # noqa: E402
from common.text import ModelResetter  # noqa: E402
from human_numbers import bs, sequence_dls, vocab  # noqa: E402


class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x
        mask = x.new_empty(*x.shape).bernoulli_(1 - self.p)
        return x * mask.div_(1 - self.p)


d = Dropout(0.5)
x = torch.ones(2, 8)
print("Dropout(0.5) in training mode:", d.train()(x).tolist(), "\n... in eval mode:", d.eval()(x)[0].tolist())


class LMModel7(nn.Module):
    def __init__(self, vocab_sz, n_hidden, n_layers, p):
        super().__init__()
        self.i_h = nn.Embedding(vocab_sz, n_hidden)
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_layers, batch_first=True)
        self.drop = nn.Dropout(p)
        self.h_o = nn.Linear(n_hidden, vocab_sz)
        self.h_o.weight = self.i_h.weight                          # weight tying
        self.h = [torch.zeros(n_layers, bs, n_hidden) for _ in range(2)]

    def forward(self, x):
        raw, h = self.rnn(self.i_h(x), [t.to(x.device) for t in self.h])
        out = self.drop(raw)
        self.h = [h_.detach() for h_ in h]
        return self.h_o(out), raw, out

    def reset(self):
        for h in self.h:
            h.zero_()


class RNNRegularizer(Callback):
    """The model returns (prediction, raw outputs, dropped-out outputs): keep the prediction for the loss
    and the metrics, and add AR and TAR to the training loss."""

    def __init__(self, alpha=2.0, beta=1.0):
        self.alpha, self.beta = alpha, beta

    def after_pred(self):
        self.learn.pred, self.raw_out, self.out = self.learn.pred

    def after_loss(self):
        if not self.learn.training:
            return
        ar = self.out.float().pow(2).mean()
        tar = (self.raw_out[:, 1:] - self.raw_out[:, :-1]).float().pow(2).mean()
        self.learn.loss = self.learn.loss + self.alpha * ar + self.beta * tar


set_seed(42)
dls = sequence_dls(16)
learn = Learner(dls, LMModel7(len(vocab), 64, 2, 0.5), loss_func=CrossEntropyLossFlat(), metrics=accuracy,
                cbs=[ModelResetter(), RNNRegularizer(alpha=2, beta=1)])
learn.fit_one_cycle(15, 1e-2, wd=0.1)
print("Dropout + AR/TAR + weight tying + weight decay: better than the plain LSTM, and less overfitting.")
