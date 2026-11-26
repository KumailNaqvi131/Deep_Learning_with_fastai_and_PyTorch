"""Shared by the Chapter 11 scripts: the tokenizer and numericalizer written as Transforms
(fastai's Tokenizer.from_folder(path) and Numericalize())."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.midlevel import Transform  # noqa: E402
from common.text import Numericalize as _Num  # noqa: E402
from common.text import Tokenizer as _Tok  # noqa: E402
from common.text import detokenize  # noqa: E402


class Tokenizer(Transform):
    """Path or str -> list of tokens. decodes: tokens -> readable text."""

    def __init__(self):
        super().__init__()
        self.tok, self.name = _Tok(), "Tokenizer"

    def encodes(self, x: (str, Path)):
        return self.tok(x.read_text(encoding="utf-8") if isinstance(x, Path) else x)

    def decodes(self, x: list):
        return detokenize(x)


class Numericalize(Transform):
    """list of tokens -> tensor of ids. setups builds the vocab from the (training) token lists."""

    def __init__(self, min_freq=3):
        super().__init__()
        self.min_freq, self.name = min_freq, "Numericalize"

    def setups(self, items):
        self.num = _Num(min_freq=self.min_freq).setup(items)
        self.vocab = self.num.vocab

    def encodes(self, x: list):
        return self.num(x)

    def decodes(self, x):
        return self.num.decode(x)
