"""
Chapter 10 — NLP Deep Dive: RNNs
Topic 5: Training a text classifier, step 1: fine-tuning the language model (book sections
"Language Model Using DataBlock", "Fine-Tuning the Language Model", "Saving and Loading Models")

ULMFiT (Universal Language Model Fine-tuning, Howard & Ruder 2018), three steps:
  1. a language model pretrained on a big general corpus (Wikipedia)          -> given
  2. fine-tune it on the texts of our task (all IMDb reviews, labelled or not) -> this script
  3. fine-tune a classifier that uses this language model's encoder            -> 08_*.py
Step 2 teaches the model the style and vocabulary of movie reviews.

    learn = language_model_learner(dls_lm, AWD_LSTM, drop_mult=0.3,
                                   metrics=[accuracy, Perplexity()]).to_fp16()
    learn.fit_one_cycle(1, 2e-2)        # frozen: only the embeddings (new words!) are trained
    learn.save('1epoch')
    learn.unfreeze()
    learn.fit_one_cycle(10, 2e-3)
    learn.save_encoder('finetuned')     # the model without its final layer (the "decoder")

Run:
    python 05_fine_tune_language_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import accuracy, set_seed  # noqa: E402
from common.text import language_model_learner, perplexity, pretrained_wikitext_lm  # noqa: E402
from imdb_lm import CONFIG, OUT, get_dls_lm  # noqa: E402

set_seed(42)
pretrained = pretrained_wikitext_lm(CONFIG)          # step 1 (our stand-in for WikiText-103)
dls_lm = get_dls_lm()
print(f"vocab: {len(dls_lm.vocab)} tokens; {len(dls_lm.train)} training batches of 128 x 80 tokens")
dls_lm.show_batch(2)

learn = language_model_learner(dls_lm, drop_mult=0.3, config=CONFIG, pretrained_fname=pretrained,
                               metrics=[accuracy, perplexity], path=OUT).to_fp16()
print("\nFrozen: training only the embeddings and the decoder")
learn.fit_one_cycle(1, 2e-2)
learn.save("1epoch")
learn = learn.load("1epoch")                            # (how you resume later)

print("\nUnfrozen: training the whole model")
learn.unfreeze()
learn.fit_one_cycle(6, 2e-3)
learn.save("lm_finetuned")
print("saved:", learn.save_encoder("finetuned"))
print("accuracy = how often the model guesses the next word right; perplexity = exp(loss)")
