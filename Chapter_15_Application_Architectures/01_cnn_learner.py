"""
Chapter 15 — Application Architectures Deep Dive
Topic 1: Computer vision: cnn_learner (book section "cnn_learner")

What cnn_learner(dls, resnet34) builds:
  1. the body: the pretrained network cut just before its pooling and classifier layers
     (fastai's model_meta says where to cut: -2 for torchvision ResNets);
  2. a new head for our classes: create_head(nf, n_out):
       AdaptiveConcatPool2d   avg pool AND max pool, concatenated (2 x 512 features for a ResNet-34)
       Flatten
       BatchNorm1d, Dropout(0.25), Linear(1024, 512), ReLU
       BatchNorm1d, Dropout(0.5),  Linear(512, n_out)
  3. a splitter into parameter groups (early body, late body, head) for freeze() and
     discriminative learning rates.

Run:
    python 01_cnn_learner.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from common import AdaptiveConcatPool2d, create_body, create_head, resnet34  # noqa: E402
from common.models import cnn_splitter, load_arch  # noqa: E402

full = load_arch(resnet34, pretrained=True)
print("the last children of torchvision's resnet34:", [type(c).__name__ for c in list(full.children())[-4:]])
body = create_body(resnet34, pretrained=True)            # == nn.Sequential(*list(full.children())[:-2])
x = torch.randn(2, 3, 224, 224)
print("body output:", tuple(body(x).shape), "-> 512 channels on a 7x7 grid")

head = create_head(512, 2, ps=0.5)
print("\ncreate_head(512, 2):\n", head)

pool = AdaptiveConcatPool2d()
print("\nAdaptiveConcatPool2d:", tuple(pool(body(x)).shape), " = [max pool, avg pool]:",
      torch.allclose(pool(body(x))[:, 512:], nn.AdaptiveAvgPool2d(1)(body(x))))

model = nn.Sequential(body, head)
groups = cnn_splitter(model)
print("\nparameter groups (for freeze / discriminative learning rates):",
      [f"{sum(p.numel() for p in g):,}" for g in groups])
