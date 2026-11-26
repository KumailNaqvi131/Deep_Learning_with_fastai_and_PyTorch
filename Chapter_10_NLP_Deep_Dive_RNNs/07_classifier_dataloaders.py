"""
Chapter 10 — NLP Deep Dive: RNNs
Topic 7: Creating the classifier DataLoaders (book section of the same name)

For the classifier, each review is one item with its label (pos/neg), and:
  * we must use the language model's vocab (vocab=dls_lm.vocab), so that token i means the same
    word for the fine-tuned encoder;
  * reviews have different lengths, so each batch is padded to its longest review with xxpad
    (index 1). To waste little computation on padding, fastai puts texts of similar length in the
    same batch (sorted, with some randomness on the training set).

    dls_clas = DataBlock(
        blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab),CategoryBlock),
        get_y = parent_label,
        get_items=partial(get_text_files, folders=['train', 'test']),
        splitter=GrandparentSplitter(valid_name='test')
    ).dataloaders(path, path=path, bs=128, seq_len=72)

Run (after 05_fine_tune_language_model.py):
    python 07_classifier_dataloaders.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.text import clas_dataloaders, imdb_texts  # noqa: E402
from imdb_lm import finetuned_vocab, path  # noqa: E402

train_texts, train_labels = imdb_texts(path, "train")
valid_texts, valid_labels = imdb_texts(path, "test")
dls_clas = clas_dataloaders(train_texts, train_labels, valid_texts, valid_labels, text_vocab=finetuned_vocab(), bs=128)
dls_clas.show_batch(3)

nums_samp = [dls_clas.num(dls_clas.tok(t)) for t in train_texts[:10]]
print("\nlengths of 10 reviews (in tokens):", [len(n) for n in nums_samp])
xb, yb = dls_clas.train.one_batch()
lens = (xb != 1).sum(1)
print(f"one training batch: {tuple(xb.shape)}; texts in it have {lens.min()}..{lens.max()} tokens "
      "(similar lengths: little padding)")
print("the start of the shortest text in the batch (padding comes first):",
      " ".join(dls_clas.num.decode(xb[lens.argmin()][:12])))
