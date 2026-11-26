# Chapter 16: The Training Process

Optimizers and callbacks from the inside: a generic optimizer made of small step functions (SGD, momentum, RMSProp, Adam, weight decay), and the callback system that lets you change any part of the training loop.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Baselines: Adam vs plain SGD (ResNet-34 from scratch on Imagenette) | Establishing a Baseline | `01_establishing_a_baseline.py` |
| 2 | `Optimizer(params, cbs)` and `sgd_cb` | A Generic Optimizer | `02_generic_optimizer.py` |
| 3 | Moving averages of gradients: `average_grad`, `momentum_step` | Momentum | `03_momentum.py` |
| 4 | `average_sqr_grad`, `rms_prop_step` | RMSProp | `04_rmsprop.py` |
| 5 | Adam = momentum + RMSProp + debiasing (checked against `torch.optim.AdamW`) | Adam | `05_adam.py` |
| 6 | L2 regularization vs decoupled weight decay | Decoupled Weight Decay | `06_decoupled_weight_decay.py` |
| 7 | The events, writing callbacks, `TerminateOnNaNCallback`, `CancelBatchException`, order | Callbacks, Creating a Callback, Callback Ordering and Exceptions | `07_callbacks.py` |

`training_setup.py` holds the DataLoaders, `get_learner`, and the generic `Optimizer` with all its step callbacks.

## Run

```bash
cd Chapter_16_The_Training_Process
python 01_establishing_a_baseline.py
python 02_generic_optimizer.py
python 03_momentum.py
python 04_rmsprop.py
python 05_adam.py
python 06_decoupled_weight_decay.py
python 07_callbacks.py
```

## Notes
Our `Optimizer` subclasses `torch.optim.Optimizer` (only `step` is ours), so it works with PyTorch's mixed-precision `GradScaler`. The callback system the scripts use is in `common/learner.py`. Chapter 19 writes a smaller one from scratch.

## Key takeaways
- Every optimizer is a recipe for turning gradients into weight updates: momentum smooths them, RMSProp scales them per parameter, Adam does both.
- Decoupled weight decay (AdamW) isn't the same as L2 regularization once the optimizer has momentum or adaptive learning rates.
- Callbacks read and change the learner's state (batch, predictions, loss, gradients) and can stop any phase with an exception. That's how mixup, AR/TAR, mixed precision and the lr finder are implemented.
