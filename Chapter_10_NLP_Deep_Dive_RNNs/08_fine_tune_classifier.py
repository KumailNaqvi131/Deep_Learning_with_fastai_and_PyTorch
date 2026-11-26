"""
Chapter 10 — NLP Deep Dive: RNNs
Topic 8: Fine-tuning the classifier (book section of the same name)

Load the fine-tuned encoder into a classifier, then unfreeze gradually: first the head only,
then also the last LSTM layer (freeze_to(-2)), then one more (freeze_to(-3)), then everything,
with discriminative learning rates (each group 2.6x smaller than the next: the ULMFiT paper's
magic number):

    learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy).to_fp16()
    learn = learn.load_encoder('finetuned')
    learn.fit_one_cycle(1, 2e-2)
    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))
    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
    learn.unfreeze()
    learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))

One change from the book for the last step: with the book's discriminative learning rates
slice(1e-3/(2.6**4), 1e-3) the embedding layer gets a learning rate 45x smaller than the head, and
our classifier gets stuck around 60%. Our small pretrained model (trained on a synthetic corpus)
hasn't learned that "superb" and "awful" mean opposite things the way a WikiText-103 model has,
so the embeddings must change more: we finish with the same lr (3e-3) for all layers (~92%).
Try the book's version (BOOK_SCHEDULE = True) to see the difference.

Run (after 05_fine_tune_language_model.py):
    python 08_fine_tune_classifier.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import accuracy, set_seed  # noqa: E402
from common.text import clas_dataloaders, imdb_texts, text_classifier_learner  # noqa: E402
from imdb_lm import CONFIG, OUT, finetuned_vocab, path  # noqa: E402

set_seed(42)
train_texts, train_labels = imdb_texts(path, "train")
valid_texts, valid_labels = imdb_texts(path, "test")
dls_clas = clas_dataloaders(train_texts, train_labels, valid_texts, valid_labels, text_vocab=finetuned_vocab(), bs=128)

learn = text_classifier_learner(dls_clas, drop_mult=0.5, metrics=accuracy, config=CONFIG, path=OUT).to_fp16()
learn = learn.load_encoder("finetuned")
learn.freeze()
print("head only:")
learn.fit_one_cycle(1, 2e-2)
print("\n+ last LSTM layer:")
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2 / (2.6 ** 4), 1e-2))
print("\n+ one more layer:")
learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3 / (2.6 ** 4), 5e-3))
print("\neverything:")
learn.unfreeze()
BOOK_SCHEDULE = False
learn.fit_one_cycle(4, slice(1e-3 / (2.6 ** 4), 1e-3) if BOOK_SCHEDULE else 3e-3)

# (The synthetic reviews use a vocabulary of a few hundred words: words they never contain, such as
#  "liked", become xxunk, so try sentences made of words that appear in the reviews.)
for review in ["I loved it! The acting was superb.", "The plot was boring and the acting was awful.",
               "The ending was not bad, and the cast did a superb job."]:
    label, _, probs = learn.predict(review)
    print(f"{review!r:55} -> {label} ({probs.max():.3f})")
learn.save("classifier")
