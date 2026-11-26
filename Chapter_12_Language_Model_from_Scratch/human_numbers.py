"""Shared by the Chapter 12 scripts: the Human Numbers dataset as one stream of tokens.

    lines = train.txt + valid.txt                   # 'one', 'two', ..., 'nine thousand nine hundred ninety nine'
    text = ' . '.join([l.strip() for l in lines])   # numbers separated by ' . '
    tokens = text.split(' ')
    vocab = L(*tokens).unique()                     # 30 words
    word2idx = {w:i for i,w in enumerate(vocab)}
    nums = L(word2idx[i] for i in tokens)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import URLs, DataLoaders, untar_data  # noqa: E402

path = untar_data(URLs.HUMAN_NUMBERS)
lines = []
for fname in ("train.txt", "valid.txt"):
    with open(path / fname) as f:
        lines += f.readlines()
text = " . ".join([line.strip() for line in lines])
tokens = text.split(" ")
vocab = list(dict.fromkeys(tokens))                    # unique, in order of first appearance
word2idx = {w: i for i, w in enumerate(vocab)}
nums = [word2idx[t] for t in tokens]
bs = 64


def group_chunks(ds, bs):
    """Reorder so that batch k+1, row i continues batch k, row i (needed to keep the hidden state)."""
    m = len(ds) // bs
    new_ds = []
    for i in range(m):
        new_ds += [ds[i + m * j] for j in range(bs)]
    return new_ds


def three_word_dls():
    """(3 words -> the next word) samples, without overlap. 80% train / 20% valid, in order."""
    seqs = [(torch.tensor(nums[i:i + 3]), nums[i + 3]) for i in range(0, len(nums) - 4, 3)]
    cut = int(len(seqs) * 0.8)
    return DataLoaders(DataLoader(seqs[:cut], batch_size=bs, shuffle=False), DataLoader(seqs[cut:], batch_size=bs, shuffle=False)), seqs


def grouped_dls(seqs):
    cut = int(len(seqs) * 0.8)
    return DataLoaders(DataLoader(group_chunks(seqs[:cut], bs), batch_size=bs, shuffle=False, drop_last=True),
                       DataLoader(group_chunks(seqs[cut:], bs), batch_size=bs, shuffle=False, drop_last=True))


def sequence_dls(sl=16):
    """(sl words -> the sl next words) samples, grouped for a stateful RNN (Ch. 12 'Creating More Signal')."""
    seqs = [(torch.tensor(nums[i:i + sl]), torch.tensor(nums[i + 1:i + sl + 1])) for i in range(0, len(nums) - sl - 1, sl)]
    return grouped_dls(seqs)
