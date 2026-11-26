# Chapter 13: Convolutional Neural Networks

Convolutions from scratch, the first CNN, convolution arithmetic and receptive fields, colour images, then how to make training stable: looking inside the network with ActivationStats, bigger batches, 1cycle training and batch normalization.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Edge-detecting kernels applied by hand | The Magic of Convolutions, Mapping a Convolutional Kernel | `01_convolution_by_hand.py` |
| 2 | `F.conv2d` on a batch with 4 kernels; padding and stride; convolution = unfold + matmul | Convolutions in PyTorch, Strides and Padding, Understanding the Convolution Equations | `02_convolutions_in_pytorch.py` |
| 3 | A CNN of stride-2 convs on 3s vs 7s, `learn.summary()`, receptive fields | Our First Convolutional Neural Network ... Receptive Fields | `03_first_cnn.py` |
| 4 | Colour images: 3 input channels, the kernels of a pretrained ResNet | Color Images | `04_color_images.py` |
| 5 | All 10 digits at lr 0.06: training fails; `ActivationStats`; a bigger batch | Improving Training Stability, A Simple Baseline, Increase Batch Size | `05_baseline_and_batch_size.py` |
| 6 | `fit_one_cycle`, the lr and momentum schedule, `color_dim` | 1cycle Training | `06_one_cycle_training.py` |
| 7 | Batch normalization (checked by hand), then 5 epochs at lr 0.1 | Batch Normalization | `07_batch_normalization.py` |

`mnist_full.py` holds the full-MNIST DataLoaders and the simple CNN for scripts 5-7.

## Run

```bash
cd Chapter_13_Convolutional_Neural_Networks
python 01_convolution_by_hand.py
python 02_convolutions_in_pytorch.py
python 03_first_cnn.py
python 04_color_images.py
python 05_baseline_and_batch_size.py
python 06_one_cycle_training.py
python 07_batch_normalization.py
```

The synthetic MNIST has 60,000 training and 10,000 test digits, like the real one (`--real` downloads it). Expected progression: plain training at a high lr trains badly (the book: 11%, here 50-70% depending on the run) → bigger batch 80% → 1cycle 96% → batchnorm 98.8% → batchnorm, 5 epochs 99.9%.

## Key takeaways
- A convolution is a small set of weights slid over the image: far fewer parameters than a linear layer, and the same feature detector everywhere.
- Output size = (n + 2*pad - ks) // stride + 1. Double the channels when stride 2 halves the grid. Deeper layers see larger parts of the image.
- Look at activation statistics: activations collapsing to zero means training is going wrong.
- 1cycle (warm-up then annealing) and batch normalization make high learning rates usable and training smooth.
