"""
Chapter 1 — Your Deep Learning Journey
Topic 6: Beyond images: classifying movie reviews (book section "Deep Learning Is Not Just
for Image Classification")

The book's code:

    dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    learn.fine_tune(4, 1e-2)
    learn.predict("I really liked that movie!")

As with images, we start from a *pretrained* model: a language model (AWD-LSTM) trained on a
large general corpus (the book uses Wikipedia; here a synthetic encyclopedia-like corpus, trained
automatically the first time, ~10 s on a GPU). Chapter 10 explains every step (tokenization,
numericalization, fine-tuning the language model first: the ULMFiT approach).

Run:
    python 06_text_sentiment_imdb.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import *  # noqa: F401,F403,E402
from common.text import clas_dataloaders, imdb_texts, text_classifier_learner  # noqa: E402

set_seed(42)
CONFIG = dict(emb_sz=200, n_hid=400, n_layers=3)      # a smaller AWD-LSTM than the book's (400/1152) so it's quick

path = untar_data(URLs.IMDB)
train_texts, train_labels = imdb_texts(path, "train")   # the folder names (pos/neg) are the labels
valid_texts, valid_labels = imdb_texts(path, "test")    # valid='test'
print(f"{len(train_texts)} training reviews, {len(valid_texts)} validation reviews. Example ({train_labels[0]}):")
print(" ", train_texts[0][:300], "...\n")

dls = clas_dataloaders(train_texts, train_labels, valid_texts, valid_labels, bs=64)
dls.show_batch(max_n=3)

learn = text_classifier_learner(dls, drop_mult=0.5, metrics=accuracy, config=CONFIG, pretrained=True)
# The book uses learn.fine_tune(4, 1e-2). fine_tune gives the earliest layers a learning rate lr_mult=100
# times smaller than the head. Our small pretrained model knows much less English than the book's
# WikiText-103 model, so its early layers need to change more: lr_mult=10.
learn.fine_tune(4, 1e-2, lr_mult=10)

# The book tries "I really liked that movie!". Our synthetic reviews never use the word "liked" (it would
# become the unknown token xxunk), so we use words that appear in them:
for review in ["I loved it! The acting was superb.", "What a boring, predictable film. I walked out."]:
    label, idx, probs = learn.predict(review)
    print(f"{review!r:50} -> {label} (probabilities neg/pos: {probs[0]:.3f} / {probs[1]:.3f})")
