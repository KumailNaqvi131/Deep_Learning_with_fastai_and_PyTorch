"""
Chapter 12 — A Language Model from Scratch
Topic 1: The data (book section "The Data")

Human Numbers: the first 10,000 numbers written out in English. A tiny dataset that is easy to
understand yet a real sequence-prediction task: predict the next word of
"one . two . three . ... . nine thousand nine hundred ninety nine".
(This dataset is generated exactly like fastai's, so it is identical to the real thing.)

Run:
    python 01_the_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from human_numbers import lines, nums, path, text, tokens, vocab  # noqa: E402

print("files:", sorted(p.name for p in path.iterdir() if not p.name.startswith(".")))
print(f"{len(lines)} lines; first ones: {[line.strip() for line in lines[:3]]}, last: {lines[-1].strip()!r}")
print("\ntext[:100] =", repr(text[:100]))
print(f"\n{len(tokens)} tokens; tokens[100:110] =", tokens[100:110])
print(f"vocab ({len(vocab)} words):", vocab)
print("\nnums[:20] =", nums[:20])
print("decoded:   ", " ".join(vocab[i] for i in nums[:20]))
