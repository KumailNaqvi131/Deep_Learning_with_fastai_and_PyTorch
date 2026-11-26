# Chapter 15: Application Architectures Deep Dive

What the learners of fastai's applications build: `cnn_learner`'s body and head, `unet_learner` for segmentation, a custom Siamese model, and the text and tabular architectures.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Cutting the pretrained body, `create_head`, AdaptiveConcatPool2d, the parameter groups | cnn_learner | `01_cnn_learner.py` |
| 2 | U-Net: encoder, decoder, skip connections, upsampling; training on CamVid | unet_learner | `02_unet_learner.py` |
| 3 | A Siamese network: shared encoder, concatenated features, custom splitter, freeze / unfreeze | A Siamese Network | `03_siamese_network.py` |
| 4 | `SentenceEncoder` reading long texts in bptt chunks, masked concat pooling; `TabularModel` | Natural Language Processing, Tabular | `04_nlp_and_tabular_architectures.py` |

## Run

```bash
cd Chapter_15_Application_Architectures
python 01_cnn_learner.py
python 02_unet_learner.py
python 03_siamese_network.py
python 04_nlp_and_tabular_architectures.py
```

## Notes
- `common/models.py` contains `SimpleUnet`, a simplified version of fastai's `DynamicUnet`: a ResNet encoder, nearest-neighbour upsampling + convs, BatchNorm on the skip connections, and a final layer that also sees the input image.
- Expected results: the U-Net gets about 95% pixel accuracy on the synthetic CamVid scenes, and the Siamese model about 95% on "same breed?".

## Key takeaways
- Every application model = a body that extracts features + a head made for the task. Transfer learning replaces the head.
- A custom splitter is all it takes to use `freeze()` and discriminative learning rates with your own architectures.
- The same ideas carry over to text (encoder + pooling head) and tables (embeddings + MLP).
