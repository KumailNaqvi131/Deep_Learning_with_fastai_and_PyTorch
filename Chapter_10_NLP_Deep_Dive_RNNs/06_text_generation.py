"""
Chapter 10 — NLP Deep Dive: RNNs
Topic 6: Text generation (book section "Text Generation")

A language model can write: predict the next word, sample one from its probabilities, add it to
the text, repeat. temperature < 1 makes it pick likely words (safer, more repetitive); > 1 makes
it more random.

    TEXT = "I liked this movie because"
    N_WORDS = 40
    N_SENTENCES = 2
    preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)]

Run (after 05_fine_tune_language_model.py):
    python 06_text_generation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import set_seed  # noqa: E402
from common.text import language_model_learner  # noqa: E402
from imdb_lm import CONFIG, OUT, finetuned_vocab, get_dls_lm  # noqa: E402

set_seed(0)
dls_lm = get_dls_lm(vocab=finetuned_vocab())
learn = language_model_learner(dls_lm, config=CONFIG, pretrained=False, path=OUT).load("lm_finetuned")

TEXT = "I liked this movie because"
N_WORDS = 40
for temperature in (0.5, 0.75, 1.2):
    print(f"\ntemperature {temperature}:")
    for _ in range(2):
        print("  ", learn.predict(TEXT, N_WORDS, temperature=temperature))
print("\n", learn.predict("This film was awful.", N_WORDS, temperature=0.75))
