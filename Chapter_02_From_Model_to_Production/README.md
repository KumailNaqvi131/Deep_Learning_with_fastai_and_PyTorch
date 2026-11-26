# Chapter 2: From Model to Production

The whole life of a project: collect data, check and clean it, build DataLoaders, augment, train, find bad data with the model, export, run inference in an app, and deploy carefully.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Gathering data, `download_images`, finding and deleting broken files with `verify_images` | Gathering Data | `01_gathering_data.py` |
| 2 | `DataBlock`: blocks, get_items, splitter, get_y; Resize crop / squish / pad, RandomResizedCrop | From Data to DataLoaders | `02_from_data_to_dataloaders.py` |
| 3 | `aug_transforms`: batch augmentation on the GPU | Data Augmentation | `03_data_augmentation.py` |
| 4 | Training, confusion matrix, top losses, and a CSV "cleaner" to fix the data | Training Your Model, and Using It to Clean Your Data | `04_train_and_clean_data.py` |
| 5 | `learn.export()`, `load_learner`, `predict` | Using the Model for Inference | `05_export_and_inference.py` |
| 6 | A web app (standard library only): upload a picture, get a prediction | Creating a Notebook App, Turning Your Notebook into a Real App, Deploying | `06_web_app.py` |
| 7 | Out-of-domain data and domain shift: confident wrong answers | How to Avoid Disaster, Unforeseen Consequences and Feedback Loops | `07_how_to_avoid_disaster.py` |

## Run

```bash
cd Chapter_02_From_Model_to_Production
python 01_gathering_data.py
python 02_from_data_to_dataloaders.py
python 03_data_augmentation.py
python 04_train_and_clean_data.py             # then edit outputs/cleaner.csv and run with --apply
python 05_export_and_inference.py
python 06_web_app.py                          # open http://localhost:8000  (--selftest to just test it)
python 07_how_to_avoid_disaster.py
```

## Data
The book downloads bear images with a Bing image search, which needs an API key. Here a synthetic bears dataset is generated with the same layout (`bears/<grizzly|black|teddy>/*.jpg`). Like real search results, it contains broken downloads, a few mislabelled pictures (teddies filed as grizzlies) and two pictures with no bear at all. Script 4 finds them. To use real photos, run `01_gathering_data.py --urls <folder of grizzly.txt, black.txt, teddy.txt URL lists>`, or put images in `data/real/bears/<class>/`, and add `--real`.

## Notes and changes from the book
- `ImageClassifierCleaner` and the Voilà app are notebook widgets. Here the cleaner writes `outputs/cleaner.csv` (with suggested actions to edit), and the app is a small `http.server`.
- `export()` pickles the model *and* the DataBlock. As in fastai, functions used in the DataBlock must be importable (no lambdas).

## Key takeaways
- Train a model early: it helps you find data problems. Clean the data, then retrain.
- `item_tfms` run on each image on the CPU (so items can be batched). `batch_tfms` run on the GPU on whole batches.
- A model gives a confident answer even for things it has never seen. Roll out gradually, keep humans in the loop, and monitor.
