"""Shared code for the chapter scripts: a small, readable re-implementation of the parts of fastai
the book uses (plain PyTorch underneath), plus offline datasets.

Every chapter script starts with

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

so that `from common import ...` works whatever folder you run it from. Then

    from common import *            # the book's `from fastai.vision.all import *`

gives you the same names the book uses (untar_data, URLs, DataBlock, Learner, ...).
The text, tabular and collaborative-filtering parts live in common.text, common.tabular and
common.collab and are imported explicitly where needed.
"""

from functools import partial  # noqa: F401  (the book uses partial a lot)
from pathlib import Path  # noqa: F401

import numpy as np  # noqa: F401
import torch  # noqa: F401
import torch.nn.functional as F  # noqa: F401
from torch import nn, tensor  # noqa: F401
from torchvision.models import resnet18, resnet34, resnet50  # noqa: F401

from .data import URLs, untar_data  # noqa: F401
from .hooks import ActivationStats, Hook, Hooks  # noqa: F401
from .learner import (SGD, Adam, BCEWithLogitsLossFlat, Callback, CancelBatchException,  # noqa: F401
                      CancelEpochException, CancelFitException, CancelTrainException, CancelValidException,
                      ClassificationInterpretation, CrossEntropyLossFlat, DataLoaders, DeviceDataLoader,
                      LabelSmoothingCrossEntropy, Learner, MSELossFlat, accuracy, accuracy_multi, error_rate,
                      foreground_acc, load_learner, mse, rmse)
from .models import (AdaptiveConcatPool2d, ConvLayer, Embedding, Flatten, ResBlock, SigmoidRange, create_body,  # noqa: F401
                     create_head, sigmoid_range, xresnet18, xresnet34, xresnet50)
from .utils import (BOOK_ROOT, DATA_ROOT, chapter_outputs, count_parameters, get_device, outputs_dir,  # noqa: F401
                    set_seed, show_images, try_import, try_import_matplotlib, use_real_data)
from .vision import (ColReader, ColSplitter, CategoryBlock, DataBlock, FuncSplitter, GrandparentSplitter,  # noqa: F401
                     ImageBlock, ImageDataLoaders, IndexSplitter, MaskBlock, MultiCategoryBlock, Normalize,
                     PILImage, PILImageBW, PointBlock, RandomResizedCrop, RandomSplitter, RegexLabeller,
                     RegressionBlock, Resize, ResizeMethod, SegmentationDataLoaders, aug_transforms, cnn_learner,
                     get_image_files, imagenet_stats, parent_label, unet_learner, using_attr, verify_images,
                     vision_learner)
