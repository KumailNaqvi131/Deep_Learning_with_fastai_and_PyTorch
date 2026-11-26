"""
Chapter 10 — NLP Deep Dive: RNNs
Topic 9: Disinformation and language models (book section of the same name)

Language models can generate text that looks like real comments, at scale: the book tells how
millions of the comments sent to the FCC about net neutrality were generated. Telling generated
text from human text is hard, and an arms race: any detector can be used to train a better
generator.

A small experiment: generate fake "reviews" with our fine-tuned language model, then compare how
*surprising* each text is to the model (its perplexity). Text sampled from a model at low
temperature is less surprising to that model than human text, which is the idea behind some
detectors... but at a higher temperature the gap closes: such detectors are easy to fool.

Run (after 05_fine_tune_language_model.py):
    python 09_disinformation_and_language_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import set_seed  # noqa: E402
from common.text import imdb_texts, language_model_learner  # noqa: E402
from imdb_lm import CONFIG, OUT, finetuned_vocab, get_dls_lm, path  # noqa: E402

set_seed(1)
dls_lm = get_dls_lm(vocab=finetuned_vocab())
learn = language_model_learner(dls_lm, config=CONFIG, pretrained=False, path=OUT).load("lm_finetuned")
model = learn.model.eval()


def text_perplexity(text):
    ids = dls_lm.num(dls_lm.tok(text)).to(dls_lm.device)
    model.reset()
    with torch.no_grad():
        logits = model(ids[None, :-1])[0]
    return torch.exp(F.cross_entropy(logits.float(), ids[1:])).item()


human, _ = imdb_texts(path, "test")
human = human[:20]
print("Some generated reviews:")
fakes = {}
for temp in (0.5, 1.0):
    fakes[temp] = [learn.predict("xxbos", 60, temperature=temp) for _ in range(20)]
    print(f"  (temperature {temp})", fakes[temp][0][:200], "...")

print(f"\naverage perplexity, human reviews:            {sum(map(text_perplexity, human)) / 20:.2f}")
for temp in (0.5, 1.0):
    print(f"average perplexity, generated (temp {temp}):    {sum(map(text_perplexity, fakes[temp])) / 20:.2f}")
print("\nLow-temperature text is easy to spot; at temperature 1 the fakes look statistically like human text.")
