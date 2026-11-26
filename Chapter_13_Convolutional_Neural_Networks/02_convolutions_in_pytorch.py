"""
Chapter 13 — Convolutional Neural Networks
Topic 2: Convolutions in PyTorch; strides and padding; the convolution equations (book sections
of the same names)

F.conv2d(input, weight) works on whole batches and many kernels at once:
    input:  [batch size, channels in, height, width]
    weight: [channels out, channels in, kernel height, kernel width]
Padding adds zeros around the image so the output can keep the input's size; a stride of 2
moves the kernel 2 pixels at a time, halving the output size. Output size:
    (n + 2*pad - ks) // stride + 1
Under the hood a convolution is a matrix multiplication with many zeros and shared weights:
we check that by computing one with unfold (which extracts every patch) and @.

Run:
    python 02_convolutions_in_pytorch.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from common import ImageDataLoaders, URLs, outputs_dir, show_images, untar_data  # noqa: E402

OUT = outputs_dir(__file__)
top_edge = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]).float()
left_edge = torch.tensor([[-1, 1, 0], [-1, 1, 0], [-1, 1, 0]]).float()
diag1_edge = torch.tensor([[0, -1, 1], [-1, 1, 0], [1, 0, 0]]).float()
diag2_edge = torch.tensor([[1, -1, 0], [0, 1, -1], [0, 0, 1]]).float()
edge_kernels = torch.stack([left_edge, top_edge, diag1_edge, diag2_edge])
print("edge_kernels:", tuple(edge_kernels.shape), "-> unsqueeze(1) for 1 input channel:", tuple(edge_kernels.unsqueeze(1).shape))

path = untar_data(URLs.MNIST_SAMPLE)
dls = ImageDataLoaders.from_folder(path)
xb, yb = dls.one_batch()
xb = xb[:, :1].cpu()                        # grey-scale: keep 1 channel
print("a batch of images:", tuple(xb.shape))
batch_features = F.conv2d(xb, edge_kernels.unsqueeze(1))
print("F.conv2d(xb, edge_kernels):", tuple(batch_features.shape), "= 64 images x 4 features x 26 x 26")
f = batch_features[0]
show_images([xb[0, 0]] + [(k - k.min()) / (k.max() - k.min()) for k in f], ["image", "left", "top", "diag1", "diag2"],
            OUT / "edge_features.png", ncols=5, cell=110)

# Strides and padding
print("\noutput size (n + 2*pad - ks)//stride + 1 for a 28x28 image:")
for ks, pad, stride in [(3, 0, 1), (3, 1, 1), (3, 1, 2), (5, 2, 2), (5, 2, 1)]:
    out = F.conv2d(xb[:1], torch.randn(1, 1, ks, ks), padding=pad, stride=stride)
    print(f"   ks={ks} padding={pad} stride={stride}: {tuple(out.shape[-2:])}  formula: {(28 + 2 * pad - ks) // stride + 1}")

# A convolution is a matrix multiplication: extract every 3x3 patch, multiply by the flattened kernels
patches = F.unfold(xb[:1], kernel_size=3)                          # [1, 9, 26*26]
as_matmul = (edge_kernels.view(4, 9) @ patches).view(1, 4, 26, 26)
print("\nunfold + matrix multiply == F.conv2d:", torch.allclose(as_matmul, batch_features[:1], atol=1e-5))
