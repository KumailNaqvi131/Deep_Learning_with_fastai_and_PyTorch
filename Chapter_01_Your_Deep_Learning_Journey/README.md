# Chapter 1: Your Deep Learning Journey

What deep learning is, what you need to start, and a tour of what it can do: images, segmentation, text, tables and recommendations, each in a few lines of code.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Checking your setup (PyTorch, GPU, optional packages) | The Software: PyTorch, fastai, and Jupyter | `01_check_setup.py` |
| 2 | Your first model: cat or dog? `fine_tune` a pretrained ResNet-34 | Your First Model, How Our Image Recognizer Works | `02_first_model_cats_vs_dogs.py` |
| 3 | Samuel's machine learning loop by hand: architecture, parameters, loss, metric, validation | What Is Machine Learning?, What Is a Neural Network?, Jargon, Limitations | `03_what_is_machine_learning.py` |
| 4 | Sounds turned into spectrogram images, classified with an image model | Image Recognizers Can Tackle Non-Image Tasks | `04_non_image_tasks_as_images.py` |
| 5 | Segmentation of street scenes with a U-Net (CamVid) | Deep Learning Is Not Just for Image Classification | `05_segmentation_camvid.py` |
| 6 | Sentiment of movie reviews (IMDb) with a pretrained language model | Deep Learning Is Not Just for Image Classification | `06_text_sentiment_imdb.py` |
| 7 | Tabular data: income from census data | Deep Learning Is Not Just for Image Classification | `07_tabular_adult_income.py` |
| 8 | Recommendations: collaborative filtering on MovieLens | Deep Learning Is Not Just for Image Classification | `08_collab_movielens.py` |
| 9 | Overfitting, validation and test sets, time-based splits | Validation Sets and Test Sets, Use Judgment in Defining Test Sets | `09_validation_and_test_sets.py` |

## Run

```bash
cd Chapter_01_Your_Deep_Learning_Journey
python 01_check_setup.py
python 02_first_model_cats_vs_dogs.py              # --image my_cat.jpg to classify your own picture
python 03_what_is_machine_learning.py
python 04_non_image_tasks_as_images.py
python 05_segmentation_camvid.py
python 06_text_sentiment_imdb.py
python 07_tabular_adult_income.py
python 08_collab_movielens.py
python 09_validation_and_test_sets.py
```

Add `--real` to scripts 2, 5, 6, 7 and 8 to download the book's real datasets.

## Notes and changes from the book
- `cnn_learner`, `unet_learner`, `fine_tune`, and the others come from `../common`, a small re-implementation of fastai in plain PyTorch with the same names and defaults.
- The text model is a smaller AWD-LSTM (200/400 instead of 400/1152), pretrained on a synthetic corpus instead of WikiText-103. Because that model knows much less English, `06` calls `fine_tune(4, 1e-2, lr_mult=10)` (the book uses the default `lr_mult=100`) so the early layers can change more. The example reviews use words that appear in the synthetic reviews.

## Key takeaways
- A pretrained model plus `fine_tune` gets excellent results with little data and little time (transfer learning).
- Model = architecture + parameters. Training means changing the parameters to lower the loss. The metric is for people.
- Always keep a validation set the model never trains on. Pick it to look like the data the model will see in production (for time series, that means the most recent period).
- The same recipe works for segmentation, text, tables and recommendations, and even for sounds turned into pictures.
