"""
Chapter 11 — Data Munging with fastai's Mid-Level API
Topic 2: Pipeline (book section "Pipeline")

A Pipeline composes several transforms: calling it runs them in order, decode runs their
decodes in reverse order, so you get back something readable.

    tfms = Pipeline([tok, num])
    t = tfms(txts[0]); t[:20]
    tfms.decode(t)[:100]

Run:
    python 02_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import URLs, untar_data  # noqa: E402
from common.midlevel import Pipeline  # noqa: E402
from common.text import get_text_files  # noqa: E402
from text_transforms import Numericalize, Tokenizer  # noqa: E402

path = untar_data(URLs.IMDB)
txts = [o.read_text() for o in get_text_files(path, folders=["train", "test"])[:2000]]

tfms = Pipeline([Tokenizer(), Numericalize()])
tfms.setup(txts)                     # Numericalize is set up on the *tokenized* texts
print(tfms)
t = tfms(txts[0])
print("tfms(txts[0])[:20] =", t[:20].tolist())
print("tfms.decode(t)[:100] =", repr(tfms.decode(t)[:100]))
