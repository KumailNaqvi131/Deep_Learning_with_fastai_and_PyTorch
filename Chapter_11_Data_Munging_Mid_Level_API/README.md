# Chapter 11: Data Munging with fastai's Mid-Level API

What's under the data block API: Transforms (encode, set up, decode), Pipelines, TfmdLists and Datasets, and how to use them for data the high-level API can't describe: pairs of images.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Tokenizer and Numericalize as Transforms; `setup`, `decode`; type dispatch; `NormalizeMean` | Going Deeper into fastai's Layered API, Transforms, Writing Your Own Transform | `01_transforms.py` |
| 2 | `Pipeline([tok, num])` and decoding | Pipeline | `02_pipeline.py` |
| 3 | `TfmdLists` with splits, `Categorize`, `Datasets`, `dataloaders(before_batch=pad_input)` | TfmdLists and Datasets: Transformed Collections | `03_tfmdlists_and_datasets.py` |
| 4 | `SiameseImage` and `SiameseTransform`: pairs of pet images, same breed or not | Applying the Mid-Level Data API: SiamesePair | `04_siamese_pair.py` |

`text_transforms.py` wraps the tokenizer and numericalizer of `common/text.py` as Transforms. The Transform, Pipeline, TfmdLists and Datasets classes are in `common/midlevel.py` (about 200 lines).

## Run

```bash
cd Chapter_11_Data_Munging_Mid_Level_API
python 01_transforms.py
python 02_pipeline.py
python 03_tfmdlists_and_datasets.py
python 04_siamese_pair.py
```

## Key takeaways
- A Transform encodes, can be set up on the training data, and can decode for display. Applied to a tuple, it transforms each element. With a type annotation, it only touches that type.
- Pipelines compose transforms. TfmdLists apply a pipeline lazily to items and set it up on the training split only. Datasets run several pipelines in parallel (inputs and targets).
- When the data block API doesn't fit, a custom Transform plus TfmdLists usually takes a dozen lines. Chapter 15 trains a Siamese model on these pairs.
