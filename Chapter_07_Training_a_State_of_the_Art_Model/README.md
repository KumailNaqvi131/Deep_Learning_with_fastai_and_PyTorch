# Chapter 7: Training a State-of-the-Art Model

Training from scratch on Imagenette with the techniques used by state-of-the-art image models. Each script adds one technique to the same baseline (xResNet-50, 5 epochs), so you can compare them.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Imagenette and an xResNet-50 baseline trained from scratch | Imagenette | `01_imagenette_baseline.py` |
| 2 | Normalizing with ImageNet statistics | Normalization | `02_normalization.py` |
| 3 | Small images first, then bigger ones | Progressive Resizing | `03_progressive_resizing.py` |
| 4 | `learn.tta()` | Test Time Augmentation | `04_test_time_augmentation.py` |
| 5 | Mixup as a callback (mixed images and mixed targets) | Mixup | `05_mixup.py` |
| 6 | `LabelSmoothingCrossEntropy` | Label Smoothing | `06_label_smoothing.py` |

`imagenette_data.py` holds the shared `get_dls(bs, size)`.

## Run

```bash
cd Chapter_07_Training_a_State_of_the_Art_Model
python 01_imagenette_baseline.py
python 02_normalization.py
python 03_progressive_resizing.py
python 04_test_time_augmentation.py
python 05_mixup.py
python 06_label_smoothing.py
```

## Data and changes from the book
The book uses full-size Imagenette (presize 460, train at 224). Here we use the 160px version (presize 160, train at 128), generated synthetically: 10 classes (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute), 260 training and 100 validation images per class, with smaller distractor objects from other classes in many pictures. `--real` downloads the real imagenette2-160 (95 MB). The baseline gets about 83% in 5 epochs, close to the book's 82%. Progressive resizing gets about 88%. Mixup and label smoothing don't help in 5 epochs: as the book says, they pay off on long training runs.

## Key takeaways
- Normalize inputs. It's essential when using pretrained models.
- Progressive resizing: faster training, and the size change itself works like augmentation.
- TTA improves predictions without retraining, at the cost of slower inference.
- Mixup and label smoothing reduce over-confidence. They need many epochs to show their benefit.
