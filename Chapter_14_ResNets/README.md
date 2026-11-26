# Chapter 14: ResNets

Building a modern CNN for Imagenette: fully convolutional networks, residual blocks with skip connections, the ResNet stem and stages, and bottleneck blocks for very deep networks.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Adaptive average pooling: a network that works for any image size | Going Back to Imagenette | `01_fully_convolutional_networks.py` |
| 2 | `ResBlock`: x + convs(x), the identity path for stride/channel changes, zero-init BatchNorm | Building a Modern CNN: ResNet, Skip Connections | `02_skip_connections.py` |
| 3 | The stem, 4 stages, `ResNet(n_out, [2,2,2,2])` (ResNet-18) | A State-of-the-Art ResNet | `03_state_of_the_art_resnet.py` |
| 4 | Bottleneck blocks (1x1, 3x3, 1x1), ResNet-50 | Bottleneck Layers | `04_bottleneck_layers.py` |

`resnet_data.py` holds the shared Imagenette DataLoaders and `get_learner`.

## Run

```bash
cd Chapter_14_ResNets
python 01_fully_convolutional_networks.py
python 02_skip_connections.py
python 03_state_of_the_art_resnet.py
python 04_bottleneck_layers.py
```

## Notes and changes from the book
- Batch size 64 instead of 128: our synthetic Imagenette has 2,600 training images instead of 9,469, so this keeps the number of optimizer steps closer to the book's.
- `02` compares the zero-initialised BatchNorm (fastai's `NormType.BatchZero`, each block starts as the identity) with the usual init. On this small dataset and a 5-epoch schedule, the zero-init version starts slowly and ends *behind* the plain CNN, while the normally initialised ResNet beats it clearly (about 75% vs 60%). Zero-init's benefit is stability in very deep networks.
- `04` trains ResNet-50 for 5 epochs at 128px (the book: 20 epochs at 224px). Expected accuracies: plain CNN about 60%, ResNet-18 about 80%, ResNet-50 about 81%.

## Key takeaways
- Fully convolutional networks end with adaptive average pooling, so the input size is free.
- Skip connections make it easy for a block to do nothing, so adding layers can't make the network worse to train. That's what made 100+ layer networks possible.
- Put the computation where it matters: a cheap stem with plain convs for the large early activations, then residual stages.
- Bottleneck blocks give more channels for the same cost, which is what deep ResNets use.
