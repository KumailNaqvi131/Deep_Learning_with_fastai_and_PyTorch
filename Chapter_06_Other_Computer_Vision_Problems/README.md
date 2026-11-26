# Chapter 6: Other Computer Vision Problems

Two new kinds of targets: several labels per image (multi-label classification), and numbers instead of categories (regression: the position of a face).

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | pandas basics; building a multi-label DataBlock step by step (get_x, get_y, MultiCategoryBlock, splitter) | Multi-Label Classification, The Data, Constructing a DataBlock | `01_multilabel_data_and_datablock.py` |
| 2 | One-hot targets, binary cross-entropy, `accuracy_multi`, `partial` | Binary Cross Entropy | `02_binary_cross_entropy.py` |
| 3 | Training a multi-label model; choosing the threshold | Binary Cross Entropy | `03_train_multilabel.py` |
| 4 | Head-pose regression: `get_ctr`, `PointBlock`, a person-based split, `y_range`, MSE | Regression, Assembling the Data, Training a Model | `04_regression_head_pose.py` |

## Run

```bash
cd Chapter_06_Other_Computer_Vision_Problems
python 01_multilabel_data_and_datablock.py
python 02_binary_cross_entropy.py
python 03_train_multilabel.py
python 04_regression_head_pose.py
```

## Data
- **PASCAL 2007** (synthetic): scenes with 1-3 icons of the 20 PASCAL classes, and a `train.csv` with `fname`, space-separated `labels` and `is_valid`, like the real file.
- **Biwi Kinect Head Pose** (synthetic): 24 people, 120 frames each, 640x480. Each frame comes with a `_pose.txt` file (rotation + 3D head centre) and the camera calibration `rgb.cal`, so the book's `get_ctr` works unchanged.

## Notes
- The augmentation in `common/vision.py` moves key points together with the image. `outputs/biwi_batch.png` shows the red target dot staying on the face after rotation and zoom.
- `03` picks the threshold on the validation set. That's fine here because the accuracy-vs-threshold curve is smooth.

## Key takeaways
- Multi-label = one sigmoid per class + binary cross-entropy, and a threshold to decide which labels to predict.
- Regression is "just" a different target block and loss: the data block API and `cnn_learner` stay the same. `y_range` (sigmoid_range) keeps the outputs in range.
- Split the validation set by person (or by whatever must generalise), not at random.
