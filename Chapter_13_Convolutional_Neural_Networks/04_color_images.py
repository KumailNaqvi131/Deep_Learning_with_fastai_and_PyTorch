"""
Chapter 13 — Convolutional Neural Networks
Topic 4: Color images (book section of the same name)

A colour image is a rank-3 tensor: [3 channels (red, green, blue), height, width]. A convolution on
it uses one kernel *per input channel* (a kernel is [3, 3, 3]) and adds the three results; with
ch_out output channels the weight tensor is [ch_out, ch_in, ks, ks]. The first layer of a
pretrained ResNet has 64 such 7x7x3 kernels; displayed as small images, they show edge and
colour detectors.

Run:
    python 04_color_images.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from common import URLs, get_image_files, outputs_dir, show_images, untar_data  # noqa: E402
from common.models import load_arch  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

OUT = outputs_dir(__file__)
im = Image.open(get_image_files(untar_data(URLs.PETS) / "images")[3]).convert("RGB").resize((224, 224))
t = torch.tensor(np.array(im)).permute(2, 0, 1).float() / 255
print("image tensor:", tuple(t.shape), "= channels x height x width")
show_images([im] + [t[i] for i in range(3)], ["image", "red", "green", "blue"], OUT / "channels.png", ncols=4)

conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
print("nn.Conv2d(3, 8, 3).weight.shape =", tuple(conv.weight.shape), " -> output", tuple(conv(t[None]).shape))
# one output channel = the sum over the 3 input channels of a 2D convolution each
manual = sum(F.conv2d(t[None, i:i + 1], conv.weight[:1, i:i + 1], padding=1) for i in range(3)) + conv.bias[0]
print("sum of 3 per-channel convolutions + bias == conv output channel 0:",
      torch.allclose(manual[0, 0], conv(t[None])[0, 0], atol=1e-5))

w = load_arch(resnet18, pretrained=True).conv1.weight.detach()
print("\nfirst layer of a pretrained ResNet-18:", tuple(w.shape))
show_images([(k - k.min()) / (k.max() - k.min()) for k in w[:32]], [""] * 32, OUT / "resnet_first_layer_kernels.png",
            ncols=8, cell=50)
