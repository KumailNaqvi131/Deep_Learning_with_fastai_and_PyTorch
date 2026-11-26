"""
Chapter 10 — NLP Deep Dive: RNNs
Topic 4: Putting our texts into batches for a language model (book section of the same name)

A language model predicts the next token. Texts have different lengths, so: concatenate all the
texts (shuffled each epoch) into one long stream, cut the stream into bs equal parts (one row of
the batch each), then go along the rows seq_len tokens at a time. Batch k+1 continues exactly
where batch k stopped, row by row, so the model's hidden state can carry over between batches.
The target y is the input x shifted by one token.

    dl = LMDataLoader(nums200)
    x,y = first(dl)
    x.shape, y.shape            # (torch.Size([64, 72]), torch.Size([64, 72]))

Run:
    python 04_batches_for_a_language_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.text import LMDataLoader, Numericalize, Tokenizer, get_text_files  # noqa: E402
from imdb_lm import path  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
stream = ("In this chapter, we will go back over the example of classifying movie reviews we studied in chapter 1 "
          "and dig deeper under the surface. First we will look at the processing steps necessary to convert text "
          "into numbers and how to customize it. By doing this, we'll have another example of the PreProcessor used "
          "in the data block API.\nThen we will study how we build a language model and train it for a while.")
tkn = Tokenizer()
tokens = tkn(stream)
bs, seq_len = 6, 15
d_tokens = np.array([tokens[i * seq_len:(i + 1) * seq_len] for i in range(bs)])
print("The stream cut into 6 rows of 15 tokens:")
print(pd.DataFrame(d_tokens).to_string(header=False))
bs, seq_len = 6, 5
for k in range(3):
    print(f"\nbatch {k + 1} (6 x 5): the next 5 tokens of each row")
    print(pd.DataFrame(d_tokens[:, k * seq_len:(k + 1) * seq_len]).to_string(header=False))

files = get_text_files(path, folders=["train", "test", "unsup"])
toks200 = [tkn(f.read_text()) for f in files[:200]]
num = Numericalize().setup(toks200)
nums200 = [num(t) for t in toks200]
dl = LMDataLoader(nums200, bs=64, seq_len=72)
x, y = next(iter(dl))
print(f"\nLMDataLoader on 200 reviews: x {tuple(x.shape)}, y {tuple(y.shape)}, {len(dl)} batches")
print("x[0][:20]:", " ".join(num.vocab[o] for o in x[0][:20]))
print("y[0][:20]:", " ".join(num.vocab[o] for o in y[0][:20]), " <- shifted by one token")
