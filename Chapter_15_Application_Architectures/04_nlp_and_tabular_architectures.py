"""
Chapter 15 — Application Architectures Deep Dive
Topic 4: Natural language processing and tabular architectures (book sections "Natural Language
Processing", "Tabular", "Conclusion")

  * Text classifier = the AWD-LSTM encoder of the language model + a pooling classifier head.
    Long documents are read in chunks of bptt=72 tokens (like the language model, keeping the
    hidden state between chunks), but only the outputs of the last max_len tokens are kept
    for the head, which concatenates [last hidden state, max pool, average pool] over the tokens
    that aren't padding (masked concat pooling).
  * Tabular model = an embedding per categorical column (+ embedding dropout), BatchNorm on the
    continuous columns, all concatenated, then Linear/ReLU/BatchNorm/Dropout layers.
  * The big picture: every architecture is "a body that turns the input into features + a head
    that turns features into the output"; transfer learning swaps the head and fine-tunes.

Run:
    python 04_nlp_and_tabular_architectures.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.tabular import TabularModel, emb_sz_rule  # noqa: E402
from common.text import AWD_LSTM, PoolingLinearClassifier  # noqa: E402

torch.manual_seed(0)


class SentenceEncoderChunked(torch.nn.Module):
    """fastai's SentenceEncoder: read the text bptt tokens at a time, keep outputs of the last max_len tokens."""

    def __init__(self, module, bptt=72, max_len=72 * 20, pad_idx=1):
        super().__init__()
        self.module, self.bptt, self.max_len, self.pad_idx = module, bptt, max_len, pad_idx

    def forward(self, inp):
        bs, sl = inp.shape
        self.module.reset()
        outs = []
        for i in range(0, sl, self.bptt):
            out = self.module(inp[:, i:i + self.bptt])          # the hidden state carries over to the next chunk
            if i > sl - self.max_len:
                outs.append(out)
        return inp[:, -sum(o.shape[1] for o in outs):], torch.cat(outs, dim=1)


enc = AWD_LSTM(vocab_sz=100, emb_sz=32, n_hid=64, n_layers=2).eval()
x = torch.randint(2, 100, (4, 300))                             # 4 documents of 300 tokens
x[0, :120] = 1                                                  # the first one is shorter: padding at the start
sentence_encoder = SentenceEncoderChunked(enc, bptt=72, max_len=144)
inp, out = sentence_encoder(x)
print(f"documents {tuple(x.shape)} -> read in chunks of 72 -> outputs kept for the last {out.shape[1]} tokens: {tuple(out.shape)}")
head = PoolingLinearClassifier([32 * 3, 50, 2], [0.4, 0.1]).eval()
print("masked concat pooling head output:", tuple(head((inp, out)).shape))

print("\nA tabular model for 3 categorical columns (with 10, 100 and 1000 levels) and 4 continuous ones:")
emb_szs = [(n, emb_sz_rule(n)) for n in (10, 100, 1000)]
print("embedding sizes (fastai's rule of thumb min(600, 1.6 * n**0.56)):", emb_szs)
model = TabularModel(emb_szs, n_cont=4, out_sz=1, layers=[200, 100], ps=[0.001, 0.01], embed_p=0.04)
print(model)
print("output:", tuple(model(torch.randint(0, 10, (8, 3)), torch.randn(8, 4)).shape))
