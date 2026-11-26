"""Shared by the Chapter 10 scripts: the IMDb texts, the language-model DataLoaders, and the model size.

The book uses fastai's AWD-LSTM (embedding size 400, 1152 hidden units, 3 layers) pretrained on
WikiText-103. We use a smaller AWD-LSTM (200/400/3) pretrained on our synthetic encyclopedia text
(trained automatically the first time, ~10 seconds on a GPU) so everything runs quickly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import URLs, outputs_dir, untar_data  # noqa: E402
from common.text import get_text_files, lm_dataloaders  # noqa: E402

CONFIG = dict(emb_sz=200, n_hid=400, n_layers=3)
OUT = outputs_dir(__file__)
path = untar_data(URLs.IMDB)


def all_texts():
    files = get_text_files(path, folders=["train", "test", "unsup"])   # every review: labels not needed for a LM
    return [f.read_text(encoding="utf-8") for f in files]


def get_dls_lm(vocab=None):
    """bs=128, seq_len=80 as in the book (DataBlock(blocks=TextBlock.from_folder(path, is_lm=True), ...))."""
    return lm_dataloaders(all_texts(), valid_pct=0.1, bs=128, seq_len=80, vocab=vocab, seed=42)


def finetuned_vocab():
    f = OUT / "finetuned.pth"
    if not f.exists():
        sys.exit("Run 05_fine_tune_language_model.py first (it saves the fine-tuned encoder).")
    return torch.load(f, map_location="cpu")["vocab"]
