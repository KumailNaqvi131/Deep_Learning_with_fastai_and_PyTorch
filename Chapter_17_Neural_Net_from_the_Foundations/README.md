# Chapter 17: A Neural Net from the Foundations

Only tensors: matrix multiplication with loops, then elementwise operations, broadcasting and einsum; initialization; the forward and backward passes written by hand (checked against autograd); refactoring into layer classes; and PyTorch's `autograd.Function` and `nn.Module`.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | A neuron; matrix multiplication with 3 loops vs `@` | Building a Neural Net Layer from Scratch, Modeling a Neuron, Matrix Multiplication from Scratch | `01_neuron_and_matrix_multiplication.py` |
| 2 | Elementwise arithmetic, broadcasting rules, `expand_as`, `None`/unsqueeze | Elementwise Arithmetic, Broadcasting | `02_elementwise_and_broadcasting.py` |
| 3 | `torch.einsum` | Einstein Summation | `03_einstein_summation.py` |
| 4 | `lin`, why init matters, Xavier and Kaiming init, the 2-layer model, MSE | The Forward and Backward Passes, Defining and Initializing a Layer | `04_defining_and_initializing_a_layer.py` |
| 5 | `mse_grad`, `relu_grad`, `lin_grad`, `forward_and_backward`, checked against autograd | Gradients and the Backward Pass | `05_backward_pass.py` |
| 6 | `Relu`/`Lin`/`Mse` classes, `Model`, `LayerFunction` | Refactoring the Model | `06_refactoring_the_model.py` |
| 7 | `torch.autograd.Function` (`MyRelu`), `nn.Module` with `nn.Parameter` | Going to PyTorch | `07_going_to_pytorch.py` |

## Run

```bash
cd Chapter_17_Neural_Net_from_the_Foundations
python 01_neuron_and_matrix_multiplication.py
python 02_elementwise_and_broadcasting.py
python 03_einstein_summation.py
python 04_defining_and_initializing_a_layer.py
python 05_backward_pass.py
python 06_refactoring_the_model.py
python 07_going_to_pytorch.py
```

All scripts run in seconds on a CPU and need no dataset.

## Notes
The book's `LayerFunction` version of `Lin.bwd` uses `self.inp`, which doesn't exist in that version. `06` uses the `inp` argument.

## Key takeaways
- Never loop in Python over tensor elements: use elementwise operations, broadcasting and einsum (or `@`).
- Initialization decides whether activations survive 50 layers: scale by sqrt(1/n_in) (Xavier), or sqrt(2/n_in) with ReLU (Kaiming).
- Backpropagation is the chain rule applied backwards, layer by layer: each layer turns the gradient of its output into gradients of its input and weights.
- `nn.Module` registers parameters, `autograd.Function` defines forward/backward, and PyTorch does the bookkeeping.
