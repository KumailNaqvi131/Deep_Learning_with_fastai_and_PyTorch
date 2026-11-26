# Deep Learning for Coders with fastai and PyTorch: Chapter-by-Chapter Code

Runnable, commented Python scripts for every concept in **_Deep Learning for Coders with fastai and PyTorch: AI Applications Without a PhD_** by Jeremy Howard and Sylvain Gugger (O'Reilly, 2020).

The code is written for deep learning students, professionals and self-learners:
- **One concept per file**, numbered in reading order (`01_...py`, `02_...py`, ...). The docstring at the top of each file names the book section it covers and shows the book's code.
- **Each script runs by itself** with `python <file>.py`. No Jupyter needed.
- **No fastai install needed.** The book's code uses the fastai library. Here, [`common/`](common/) re-implements the parts of fastai the book uses, in plain PyTorch and with the same names: `untar_data`, `DataBlock`, `Learner`, `fine_tune`, `lr_find`, `fit_one_cycle`, `cnn_learner`, `ClassificationInterpretation`, and so on. The book's code maps almost line for line, and you can read every piece. Part IV of the book (Chapters 17-19) builds these pieces from scratch.
- **Works offline, no big downloads.** Every dataset of the book has a synthetic look-alike with the same folder layout and file formats, generated in `data/` the first time it's needed (seconds to a minute). Add `--real` to a script (or set `REAL_DATA=1`) to download the real dataset from the same URL fastai uses (see [Data](#data)).
- **GPU when available.** Every script uses CUDA (or Apple MPS) if present and falls back to the CPU. Mixed precision (`to_fp16()`) is used where the book uses it, on NVIDIA GPUs.

## Setup

```bash
conda create -n dl4coders python=3.11 -y
conda activate dl4coders
pip install -r requirements.txt     # only torch, torchvision, numpy, pandas and pillow are required
```

For NVIDIA GPU builds of PyTorch, use the install command from https://pytorch.org/get-started/locally/. The other packages in `requirements.txt` are optional. Scripts print a message and skip the feature (plots, scikit-learn models, ...) when one is missing. Pretrained ImageNet weights (ResNet-18/34/50) are downloaded once by torchvision. Without internet, the scripts warn and continue with random weights.

Check the setup:
```bash
python Chapter_01_Your_Deep_Learning_Journey/01_check_setup.py
```

## Chapters

### Part I: Deep Learning in Practice
| Chapter | Folder | What you'll code |
|---|---|---|
| 1. Your deep learning journey | [Chapter_01_Your_Deep_Learning_Journey](Chapter_01_Your_Deep_Learning_Journey/) | Cats vs dogs in 6 lines, machine learning jargon, sounds as images, segmentation, sentiment, tabular, recommendations, validation sets |
| 2. From model to production | [Chapter_02_From_Model_to_Production](Chapter_02_From_Model_to_Production/) | Gathering and cleaning data, DataBlock, resizing, augmentation, export and inference, a web app, out-of-domain data |
| 3. Data ethics | [Chapter_03_Data_Ethics](Chapter_03_Data_Ethics/) | Simulations of representation bias, feedback loops and fairness metrics |

### Part II: Understanding fastai's Applications
| Chapter | Folder | What you'll code |
|---|---|---|
| 4. Under the hood: training a digit classifier | [Chapter_04_Training_a_Digit_Classifier](Chapter_04_Training_a_Digit_Classifier/) | Pixels, a baseline, broadcasting, SGD by hand, the MNIST loss, mini-batches, a neural net |
| 5. Image classification | [Chapter_05_Image_Classification](Chapter_05_Image_Classification/) | Pet breeds, presizing, cross-entropy, interpretation, lr finder, transfer learning, discriminative lrs |
| 6. Other computer vision problems | [Chapter_06_Other_Computer_Vision_Problems](Chapter_06_Other_Computer_Vision_Problems/) | Multi-label classification, binary cross-entropy, thresholds, regression (head pose) |
| 7. Training a state-of-the-art model | [Chapter_07_Training_a_State_of_the_Art_Model](Chapter_07_Training_a_State_of_the_Art_Model/) | Imagenette, normalization, progressive resizing, TTA, mixup, label smoothing |
| 8. Collaborative filtering deep dive | [Chapter_08_Collaborative_Filtering](Chapter_08_Collaborative_Filtering/) | Latent factors, embeddings, dot product model, weight decay, interpreting embeddings, deep collab |
| 9. Tabular modeling deep dive | [Chapter_09_Tabular_Modeling](Chapter_09_Tabular_Modeling/) | Decision trees, random forests, feature importance, partial dependence, extrapolation, neural nets, boosting |
| 10. NLP deep dive: RNNs | [Chapter_10_NLP_Deep_Dive_RNNs](Chapter_10_NLP_Deep_Dive_RNNs/) | Tokenization, subwords, numericalization, LM batches, ULMFiT fine-tuning, text generation |
| 11. Data munging with fastai's mid-level API | [Chapter_11_Data_Munging_Mid_Level_API](Chapter_11_Data_Munging_Mid_Level_API/) | Transform, Pipeline, TfmdLists, Datasets, a Siamese pair dataset |

### Part III: Foundations of Deep Learning
| Chapter | Folder | What you'll code |
|---|---|---|
| 12. A language model from scratch | [Chapter_12_Language_Model_from_Scratch](Chapter_12_Language_Model_from_Scratch/) | Human Numbers, RNNs, stateful RNNs, multilayer RNNs, LSTM from scratch, AWD-LSTM regularization |
| 13. Convolutional neural networks | [Chapter_13_Convolutional_Neural_Networks](Chapter_13_Convolutional_Neural_Networks/) | Convolutions by hand, F.conv2d, a CNN, receptive fields, ActivationStats, 1cycle, batchnorm |
| 14. ResNets | [Chapter_14_ResNets](Chapter_14_ResNets/) | Fully convolutional nets, skip connections, the ResNet stem, bottleneck layers |
| 15. Application architectures deep dive | [Chapter_15_Application_Architectures](Chapter_15_Application_Architectures/) | cnn_learner's head, U-Net, a Siamese network, text and tabular architectures |
| 16. The training process | [Chapter_16_The_Training_Process](Chapter_16_The_Training_Process/) | A generic optimizer: SGD, momentum, RMSProp, Adam, decoupled weight decay, and callbacks |

### Part IV: Deep Learning from Scratch
| Chapter | Folder | What you'll code |
|---|---|---|
| 17. A neural net from the foundations | [Chapter_17_Neural_Net_from_the_Foundations](Chapter_17_Neural_Net_from_the_Foundations/) | Matrix multiply, broadcasting, einsum, initialization, backprop by hand, autograd.Function |
| 18. CNN interpretation with CAM | [Chapter_18_CNN_Interpretation_with_CAM](Chapter_18_CNN_Interpretation_with_CAM/) | Hooks, class activation maps, Grad-CAM |
| 19. A fastai Learner from scratch | [Chapter_19_A_fastai_Learner_from_Scratch](Chapter_19_A_fastai_Learner_from_Scratch/) | Dataset, DataLoader, Module, Parameter, loss, Learner, callbacks, lr finder, 1cycle |
| 20. Concluding thoughts | [Chapter_20_Concluding_Thoughts](Chapter_20_Concluding_Thoughts/) | What to do next, and the data project checklist (no code) |

## Data

| Dataset (`URLs.X`) | Used in | Synthetic look-alike (default) | `--real` |
|---|---|---|---|
| `PETS` | 1, 5, 11, 15, 18 | 37 cartoon breeds (12 cats capitalised, 25 dogs lowercase), named `<Breed>_<n>.jpg` | 800 MB download |
| `BEARS` | 2 | grizzly / black / teddy, with broken files, mislabelled and off-topic images to clean | your own images in `data/real/bears/<class>/` (or `01_gathering_data.py --urls`) |
| `MNIST_SAMPLE`, `MNIST` | 4, 13 | hand-written-looking digits, same folders as fastai's | 3 MB / 15 MB download |
| `PASCAL_2007` | 6 | scenes with 1-3 of the 20 PASCAL object classes, `train.csv` with labels and `is_valid` | 1.6 GB download |
| `BIWI_HEAD_POSE` | 6 | 24 "people", 640x480 frames, `_pose.txt` + `rgb.cal` as in the real data | 450 MB download |
| `IMAGENETTE` | 7, 14, 16, 19 | 10 classes (tench, ..., parachute), 160 px, with distractor objects | 95 MB download (imagenette2-160) |
| `CAMVID_TINY` | 1, 15 | street scenes with per-pixel masks and the 32 CamVid codes | 2 MB download |
| `ADULT_SAMPLE` | 1 | census income table | small download |
| `ML_100k`, `ML_SAMPLE` | 1, 8 | 943 users x 1682 movies, 100k ratings, with interpretable movie "styles" | 5 MB download |
| `BULLDOZERS` | 9 | auction prices with planted important, redundant and time-dependent columns | Kaggle (see Chapter 9 README) |
| `IMDB` | 1, 10, 11 | positive/negative movie reviews (+ unlabelled ones) | 145 MB download |
| `HUMAN_NUMBERS` | 12 | generated exactly like the real dataset | tiny download |

The synthetic data keeps what matters for learning: classes that differ in shape, colour and texture, random positions and clutter, some genuinely confusable classes (e.g. Russian Blue vs British Shorthair), label noise to clean, a time trend in the tabular data, and so on. Results are in the same range as the book's, but not identical.

The language models are pretrained on a small synthetic "WikiText" corpus (trained automatically the first time, about 10 seconds on a GPU) instead of downloading fastai's WikiText-103 weights.

## Folder layout

```
Deep_Learning_for_Coders_with_fastai_and_PyTorch/
├── README.md             <- you are here
├── requirements.txt
├── run_all.py            <- runs every script as a smoke test
├── common/               <- a small, readable fastai in plain PyTorch
│   ├── data.py           <- URLs and untar_data (synthetic or --real)
│   ├── learner.py        <- Learner, callbacks, fit_one_cycle, fine_tune, lr_find, losses, metrics, interpretation
│   ├── vision.py         <- DataBlock, blocks, splitters, Resize, aug_transforms (GPU), cnn_learner, unet_learner
│   ├── models.py         <- create_body/create_head, xResNet, a U-Net, Embedding, sigmoid_range
│   ├── text.py           <- tokenizer rules, Numericalize, LM and classifier loaders, AWD-LSTM, ULMFiT
│   ├── tabular.py        <- add_datepart, Categorify/FillMissing/Normalize, TabularPandas, TabularModel
│   ├── collab.py         <- CollabDataLoaders, EmbeddingDotBias, EmbeddingNN
│   ├── midlevel.py       <- Transform, Pipeline, TfmdLists, Datasets (Chapter 11)
│   ├── hooks.py          <- Hook, Hooks, ActivationStats, model summary (Chapters 13 and 18)
│   ├── synth_vision.py   <- the synthetic image datasets
│   ├── synth_data.py     <- the synthetic tabular, ratings and text datasets
│   └── utils.py          <- device, seeding, output folders, image grids
├── data/                 <- datasets, generated automatically on first use
└── Chapter_XX_.../
    ├── README.md         <- topics, book sections, how to run, notes
    ├── 01_topic.py       <- one concept per file
    └── outputs/          <- images, plots and models written by the scripts
```

## How to study with this repo
1. Read a section of the book.
2. Open the matching `.py` file. The docstring names the section and shows the book's fastai code, and the comments explain each step.
3. Run it, then change something (learning rate, epochs, architecture, augmentation) and run it again.
4. When a script calls `learn.fit_one_cycle` or `DataBlock(...)`, read the implementation in `common/`. It's short, and Chapters 16 and 19 explain how it works.

Run everything at once:
```bash
python run_all.py          # or: python run_all.py 4 5   (only Chapters 4 and 5)
```
The whole book (120 scripts) takes about 40 minutes on a recent GPU. Chapters 1-2, 4-7 and 13-16 train image models and are slow on a CPU. Chapters 3, 8-12, 17 and 19 are fine on a CPU.
