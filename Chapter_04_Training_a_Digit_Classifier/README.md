# Chapter 4: Under the Hood: Training a Digit Classifier

From pixels to a trained neural network, written step by step, on 3s versus 7s from MNIST: tensors, broadcasting, gradients, SGD, a loss function, mini-batches, an optimizer, and nonlinearities.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | Images as numbers: arrays, tensors, a DataFrame view, ASCII art | Pixels: The Foundations of Computer Vision | `01_pixels.py` |
| 2 | A baseline: the "ideal" 3 and 7 (mean images), L1 and RMSE distances | First Try: Pixel Similarity | `02_pixel_similarity_baseline.py` |
| 3 | NumPy arrays vs tensors; broadcasting to measure accuracy on the whole validation set | NumPy Arrays and PyTorch Tensors, Computing Metrics Using Broadcasting | `03_tensors_and_broadcasting.py` |
| 4 | Gradients with autograd, the learning rate, the roller-coaster example, the 7 steps | Stochastic Gradient Descent ... Summarizing Gradient Descent | `04_stochastic_gradient_descent.py` |
| 5 | A linear model, why accuracy can't be the loss, `mnist_loss`, sigmoid | The MNIST Loss Function, Sigmoid | `05_mnist_loss_and_sigmoid.py` |
| 6 | DataLoader, the training loop by hand, `nn.Linear`, `BasicOptim`, `Learner` | SGD and Mini-Batches, Putting It All Together, Creating an Optimizer | `06_minibatches_and_training_loop.py` |
| 7 | ReLU, a 2-layer net (by hand and with `nn.Sequential`), going deeper with ResNet-18 | Adding a Nonlinearity, Going Deeper | `07_adding_a_nonlinearity.py` |

`mnist_data.py` loads the images as tensors for scripts 5-7.

## Run

```bash
cd Chapter_04_Training_a_Digit_Classifier
python 01_pixels.py
python 02_pixel_similarity_baseline.py
python 03_tensors_and_broadcasting.py
python 04_stochastic_gradient_descent.py
python 05_mnist_loss_and_sigmoid.py
python 06_minibatches_and_training_loop.py
python 07_adding_a_nonlinearity.py
```

Add `--real` to use the real MNIST sample (3 MB). The synthetic digits are rendered with a font and random rotation, shear, stroke width, blur and stray marks.

## Key takeaways
- A baseline first: the pixel-similarity model already gets about 95%. Anything you train must beat it.
- Learning = predict, compute a loss, compute gradients, step the weights against the gradient (times the learning rate), repeat.
- The loss must change smoothly when the weights change. Accuracy doesn't, so we use a smooth loss for training and accuracy for reporting.
- Mini-batches: more steps per epoch, and the GPU processes a whole batch at once.
- Linear layers + nonlinearities = a neural network, able to approximate any function. Deeper models get there with fewer parameters.
