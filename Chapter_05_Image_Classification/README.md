# Chapter 5: Image Classification

From cats vs dogs to 37 pet breeds, and the techniques that make image classifiers train well: presizing, debugging a DataBlock, cross-entropy, interpretation, the learning rate finder, transfer learning, discriminative learning rates, the number of epochs, deeper models and mixed precision.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Labels from file names with a regular expression | From Dogs and Cats to Pet Breeds | `01_pets_data_and_regex_labels.py` |
| 2 | Presizing: resize big on the CPU, augment once on the GPU (comparison picture) | Presizing | `02_presizing.py` |
| 3 | `show_batch`, `DataBlock.summary` (with the book's missing-Resize error), a quick baseline | Checking and Debugging a DataBlock | `03_checking_the_datablock.py` |
| 4 | Activations, softmax, log likelihood, log, `nn.CrossEntropyLoss` | Cross-Entropy Loss | `04_cross_entropy_loss.py` |
| 5 | Confusion matrix and `most_confused` | Model Interpretation | `05_model_interpretation.py` |
| 6 | Too high a learning rate; `lr_find` | The Learning Rate Finder | `06_learning_rate_finder.py` |
| 7 | `freeze`, `unfreeze`, what `fine_tune` does | Unfreezing and Transfer Learning | `07_unfreezing_and_transfer_learning.py` |
| 8 | `lr_max=slice(1e-6, 1e-4)`, `plot_loss` | Discriminative Learning Rates | `08_discriminative_learning_rates.py` |
| 9 | Choosing epochs, ResNet-50, `to_fp16()` | Selecting the Number of Epochs, Deeper Architectures | `09_epochs_and_deeper_architectures.py` |

`pets_data.py` holds the book's `pets` DataBlock, shared by the scripts.

## Run

```bash
cd Chapter_05_Image_Classification
python 01_pets_data_and_regex_labels.py
python 02_presizing.py
python 03_checking_the_datablock.py
python 04_cross_entropy_loss.py
python 05_model_interpretation.py
python 06_learning_rate_finder.py
python 07_unfreezing_and_transfer_learning.py
python 08_discriminative_learning_rates.py
python 09_epochs_and_deeper_architectures.py
```

## Data
Synthetic Oxford-IIIT Pets: 37 breeds × 100 images of different sizes (260-500 px). Each breed has its own fur colour, coat pattern, ears, eyes and snout, and some breeds share most of them (Russian Blue / British Shorthair, Ragdoll / Siamese, beagle / basset hound), so `most_confused` finds sensible mistakes, as in the book. With `--real` the real dataset (800 MB) is downloaded.

## Key takeaways
- Presizing keeps image quality: one interpolation for all augmentations, done on the GPU.
- Cross-entropy = log-softmax + negative log likelihood. It wants one class per image.
- Use the learning rate finder: choose the minimum/10 or the steepest point.
- Transfer learning: train the new head first (frozen body), then everything, with lower learning rates for the early layers.
- Deeper is not always better. Mixed precision makes training faster and uses less memory.
