"""
Chapter 10 — NLP Deep Dive: RNNs
Topic 1: Text preprocessing and word tokenization (book sections "Text Preprocessing",
"Tokenization", "Word Tokenization with fastai")

Text has to become numbers: 1) tokenization: split it into tokens (words, sub-words, characters);
2) numericalization: give each token an index in a vocabulary (03_numericalization.py).

fastai's tokenizer adds rules that keep information a plain split would lose, with special tokens:
    xxbos  beginning of a text           xxmaj  the next word started with a capital letter
    xxunk  unknown word                   xxup   the next word was ALL CAPS
    xxrep  a repeated character: !!!! -> xxrep 4 !
    xxwrep a repeated word: very very very -> xxwrep 3 very
Lowercasing everything (plus xxmaj/xxup) means "The" and "the" share one embedding, but the
model still knows about the capital. (The book uses spaCy for word splitting; we use a small
regular-expression tokenizer with the same behaviour on English: see common/text.py.)

Run:
    python 01_tokenization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.text import (Tokenizer, defaults_text_postproc_rules, defaults_text_proc_rules,  # noqa: E402
                         get_text_files, word_tokenize)
from imdb_lm import path  # noqa: E402

files = get_text_files(path, folders=["train", "test", "unsup"])
print(f"{len(files)} review files")
txt = files[0].read_text()
print("txt[:75] =", repr(txt[:75]))

toks = word_tokenize(txt)
print("\nword tokens:", toks[:30])
print("\nword_tokenize('The U.S. dollar $1 is $1.00.') =", word_tokenize("The U.S. dollar $1 is $1.00."),
      "\n(spaCy keeps 'U.S.' and '1.00' together; a simple regex splits them: tokenizers differ in details)")

tkn = Tokenizer()
print("\nwith fastai's rules:", tkn(txt)[:31])
print("\nthe rules applied to the text:", [f.__name__ for f in defaults_text_proc_rules])
print("the rules applied to the tokens:", [f.__name__ for f in defaults_text_postproc_rules])
print("\nTokenizer()('&copy;   Fast.ai www.fast.ai/INDEX') =", tkn("&copy;   Fast.ai www.fast.ai/INDEX"))
print("Tokenizer()('This was SO GOOD!!!! Very very very good') =", tkn("This was SO GOOD!!!! Very very very good"))
