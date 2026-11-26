"""
Chapter 10 — NLP Deep Dive: RNNs
Topic 3: Numericalization (book section "Numericalization with fastai")

Make a vocabulary from the tokens of the training texts: the special tokens first, then every
token appearing at least min_freq=3 times (most frequent first), at most max_vocab=60000.
Each token is then replaced by its index; tokens not in the vocab become xxunk (index 0).

    toks200 = txts[:200].map(tkn)
    num = Numericalize()
    num.setup(toks200)
    nums = num(toks)[:20]
    ' '.join(num.vocab[o] for o in nums)

Run:
    python 03_numericalization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.text import Numericalize, Tokenizer, get_text_files  # noqa: E402
from imdb_lm import path  # noqa: E402

files = get_text_files(path, folders=["train", "test", "unsup"])
txts = [f.read_text() for f in files[:2000]]
tkn = Tokenizer()
toks = tkn(txts[0])
toks200 = [tkn(t) for t in txts[:200]]

num = Numericalize()
num.setup(toks200)
print(f"vocab size {len(num.vocab)}; the first 20 tokens:", num.vocab[:20])
nums = num(toks)[:20]
print("\nnum(toks)[:20] =", nums.tolist())
print("decoded:", " ".join(num.vocab[o] for o in nums))

rare = Numericalize(min_freq=50).setup(toks200)
print(f"\nwith min_freq=50 the vocab has {len(rare.vocab)} tokens; the same text becomes:")
print(" ".join(rare.vocab[o] for o in rare(toks)[:20]), "  (rarer words -> xxunk)")
