# Chapter 10: NLP Deep Dive: RNNs

How the Chapter 1 sentiment classifier works, step by step, following ULMFiT: tokenize, numericalize, batch for a language model, fine-tune a pretrained language model on the reviews, then fine-tune a classifier on top of its encoder.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Word tokenization and fastai's special tokens (xxbos, xxmaj, xxup, xxrep, ...) | Text Preprocessing, Tokenization, Word Tokenization with fastai | `01_tokenization.py` |
| 2 | Subword tokenization: BPE from scratch, and sentencepiece | Subword Tokenization | `02_subword_tokenization.py` |
| 3 | `Numericalize`: the vocab, min_freq, xxunk | Numericalization with fastai | `03_numericalization.py` |
| 4 | Language-model batches: one stream, bs rows, seq_len steps | Putting Our Texts into Batches for a Language Model | `04_batches_for_a_language_model.py` |
| 5 | Fine-tuning the language model (frozen, then unfrozen), save the encoder | Training a Text Classifier, Fine-Tuning the Language Model, Saving and Loading Models | `05_fine_tune_language_model.py` |
| 6 | Generating text with temperature | Text Generation | `06_text_generation.py` |
| 7 | Classifier DataLoaders: the LM vocab, padding, sorting by length | Creating the Classifier DataLoaders | `07_classifier_dataloaders.py` |
| 8 | Gradual unfreezing: `freeze_to(-2)`, `freeze_to(-3)`, `unfreeze()` | Fine-Tuning the Classifier | `08_fine_tune_classifier.py` |
| 9 | Generated "reviews" and how hard they are to detect (perplexity) | Disinformation and Language Models | `09_disinformation_and_language_models.py` |

`imdb_lm.py` holds the shared data and model size. Run 05 before 06-09.

## Run

```bash
cd Chapter_10_NLP_Deep_Dive_RNNs
python 01_tokenization.py
python 02_subword_tokenization.py
python 03_numericalization.py
python 04_batches_for_a_language_model.py
python 05_fine_tune_language_model.py
python 06_text_generation.py
python 07_classifier_dataloaders.py
python 08_fine_tune_classifier.py
python 09_disinformation_and_language_models.py
```

## Data and changes from the book
- **IMDb** (synthetic): 6,000 labelled training reviews, 2,000 test reviews, 6,000 unlabelled ones, written from a grammar with sentiment words, negations ("not bad"), neutral sentences, CAPITALS, "!!!!" and `<br />` tags. `--real` downloads the real dataset (145 MB, 100,000 reviews).
- **Step 1 of ULMFiT** (the pretrained language model) uses a synthetic encyclopedia-style corpus instead of WikiText-103. It's trained automatically the first time and saved in `data/models/`.
- The AWD-LSTM is smaller (embedding 200, hidden 400, 3 layers; the book: 400, 1152, 3) so the chapter runs in a couple of minutes.
- The book's tokenizer uses spaCy. `common/text.py` uses a regex tokenizer with the same rules.
- `08`: with the book's last step `slice(1e-3/(2.6**4), 1e-3)`, the embedding layer learns too slowly for our small pretrained model, and accuracy stalls around 60%. The script finishes with one learning rate for all layers (about 92%). Set `BOOK_SCHEDULE = True` to try the book's version.

## Key takeaways
- Tokenization rules keep information (capitals, repetitions) while keeping the vocab small.
- A language model learns from text alone (self-supervised). Fine-tuning it on your corpus teaches it the style and vocabulary.
- Classifier = the fine-tuned encoder + a pooling head, trained with gradual unfreezing.
- Language models generate convincing text, which makes them a tool for disinformation.
