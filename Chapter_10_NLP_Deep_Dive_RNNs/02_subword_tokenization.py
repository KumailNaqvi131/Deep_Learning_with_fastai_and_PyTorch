"""
Chapter 10 — NLP Deep Dive: RNNs
Topic 2: Subword tokenization (book section "Subword Tokenization")

Word tokenization assumes spaces separate meaningful units; not true for Chinese, Japanese, or
German compounds, and it can't handle unknown words. Subword tokenization learns a vocabulary of
the most common character sequences from the corpus:
  1. start with single characters,
  2. repeatedly merge the most frequent adjacent pair into a new token (byte-pair encoding, BPE),
  3. stop when the vocabulary has the size you asked for.
A small vocab -> short tokens, long sequences. A big vocab -> whole words, short sequences but big
embedding matrices. The book uses SentencePiece (SubwordTokenizer(vocab_sz=...)); this script
implements BPE from scratch, then uses sentencepiece too if it's installed. ('▁' marks a space.)

Run:
    python 02_subword_tokenization.py
"""

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import outputs_dir  # noqa: E402
from common.text import get_text_files  # noqa: E402
from imdb_lm import path  # noqa: E402

files = get_text_files(path, folders=["train", "test", "unsup"])
txts = [f.read_text() for f in files[:2000]]
txt = txts[0]


def train_bpe(texts, vocab_sz):
    """Learn BPE merges until the vocabulary has vocab_sz tokens."""
    words = Counter(w for t in texts for w in re.findall(r"\S+", t.lower()))
    corpus = {tuple("▁" + w): c for w, c in words.items()}          # each word as a tuple of symbols
    vocab = {s for w in corpus for s in w}
    merges = []
    while len(vocab) < vocab_sz:
        pairs = Counter()
        for w, c in corpus.items():
            for a, b in zip(w, w[1:]):
                pairs[a, b] += c
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        merges.append((a, b))
        vocab.add(a + b)
        new_corpus = {}
        for w, c in corpus.items():                                   # apply the merge everywhere
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new_corpus[tuple(out)] = c
        corpus = new_corpus
    return merges


def bpe_tokenize(text, merges):
    toks = []
    for word in re.findall(r"\S+", text.lower()):
        w = list("▁" + word)
        for a, b in merges:
            i = 0
            while i < len(w) - 1:
                if w[i] == a and w[i + 1] == b:
                    w[i:i + 2] = [a + b]
                else:
                    i += 1
        toks += w
    return toks


for sz in (60, 150, 400):
    merges = train_bpe(txts[:500], sz)
    toks = bpe_tokenize(txt, merges)
    print(f"BPE, vocab {sz:4d}: {len(toks):4d} tokens:", " ".join(toks[:24]))

try:
    import sentencepiece as spm
    corpus_file = outputs_dir(__file__) / "texts.txt"
    corpus_file.write_text("\n".join(txts))
    for sz in (200, 500):
        prefix = str(outputs_dir(__file__) / f"spm{sz}")
        spm.SentencePieceTrainer.train(input=str(corpus_file), model_prefix=prefix, vocab_size=sz,
                                       character_coverage=1.0, model_type="unigram", minloglevel=2)
        sp = spm.SentencePieceProcessor(model_file=prefix + ".model")
        print(f"sentencepiece, vocab {sz}:", " ".join(sp.encode(txt, out_type=str)[:24]))
except ImportError:
    print("[info] sentencepiece not installed (pip install sentencepiece) -> skipped")
