# Chapter 19: A fastai Learner from Scratch

Rebuild the training stack with nothing but tensors and `F.conv2d`: Dataset, DataLoader, normalization, Module and Parameter, layers, cross-entropy with the LogSumExp trick, an SGD optimizer, a Learner with callbacks, a learning rate finder and 1cycle scheduling.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | `Dataset`, `collate`, a parallel `DataLoader`, `Normalize`, `tfm_x` | Data, Dataset | `01_data_and_dataset.py` |
| 2 | `Parameter`, `Module` (registering parameters and children), `ConvLayer`, `Linear`, `Sequential`, hooks, `simple_cnn` | Module and Parameter, Simple CNN | `02_module_and_parameter.py` |
| 3 | `nll`, `log_softmax`, `logsumexp`, `cross_entropy` | Loss | `03_loss.py` |
| 4 | `SGD`, `DataLoaders`, `Learner`, `Callback`, `SetupLearnerCB`, `TrackResults` | Learner, Callbacks | `04_learner_and_callbacks.py` |
| 5 | `LRFinder` and `OneCycle` callbacks | Scheduling the Learning Rate | `05_scheduling_the_learning_rate.py` |

Each script defines its pieces at the top level and runs a demo under `if __name__ == "__main__":`, so later scripts import earlier ones (`importlib.import_module("01_data_and_dataset")`) instead of repeating code.

## Run

```bash
cd Chapter_19_A_fastai_Learner_from_Scratch
python 01_data_and_dataset.py
python 02_module_and_parameter.py
python 03_loss.py
python 04_learner_and_callbacks.py
python 05_scheduling_the_learning_rate.py
```

## Notes and changes from the book
- The DataLoader uses a `ThreadPoolExecutor` instead of the book's `ProcessPoolExecutor`. Forking processes after PyTorch has started its CPU threads can deadlock, and Windows/macOS don't fork. Pillow releases the GIL while decoding images, so threads give real parallelism here.
- Batch size 32 instead of 128 (the synthetic Imagenette has 3.6x fewer images, so this keeps the number of SGD steps similar). With plain SGD, no momentum and no batchnorm, the tiny CNN reaches about 15% after 1 epoch and 35-40% after 8 epochs of 1cycle. Compare with what fastai's tools achieved in Chapters 14 and 16: that gap is what momentum, batchnorm, ResNets and Adam buy you.

## Key takeaways
- None of it is magic: a Learner is a loop over epochs and batches that calls the model, the loss, `backward`, the optimizer, and callbacks at each step.
- `nn.Module`'s job is bookkeeping: finding parameters recursively and passing the training flag down.
- Numerical stability matters: use the LogSumExp trick for log-softmax.
- Schedules (lr finder, 1cycle) are just callbacks that set the learning rate before each batch.
