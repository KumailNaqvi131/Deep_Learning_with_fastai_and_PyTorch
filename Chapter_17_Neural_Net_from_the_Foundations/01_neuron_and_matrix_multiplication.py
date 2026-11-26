"""
Chapter 17 — A Neural Net from the Foundations
Topic 1: Modeling a neuron; matrix multiplication from scratch (book sections "Building a
Neural Net Layer from Scratch", "Modeling a Neuron", "Matrix Multiplication from Scratch")

A neuron: output = sum of (inputs * weights) + bias, then a nonlinearity (ReLU). A layer of
neurons on a batch of inputs is a matrix multiplication. Written with three Python loops it is
painfully slow; PyTorch's @ runs optimised C/CUDA code: thousands of times faster.

    def matmul(a,b):
        (ar,ac),(br,bc) = a.shape,b.shape
        c = torch.zeros(ar, bc)
        for i in range(ar):
            for j in range(bc):
                for k in range(ac): c[i,j] += a[i,k] * b[k,j]
        return c

Run:
    python 01_neuron_and_matrix_multiplication.py
"""

import time

import torch

torch.manual_seed(0)


def relu(x):
    return x if x >= 0 else 0


inputs, weights, bias = [0.5, -1.0, 2.0], [0.1, 0.2, 0.3], 0.05
output = sum(x * w for x, w in zip(inputs, weights)) + bias
print(f"one neuron: sum(x*w) + b = {output:.3f} -> relu -> {relu(output):.3f}")


def matmul(a, b):
    (ar, ac), (br, bc) = a.shape, b.shape
    assert ac == br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                c[i, j] += a[i, k] * b[k, j]
    return c


m1 = torch.randn(5, 28 * 28)          # 5 flattened MNIST-sized images
m2 = torch.randn(784, 10)             # the weights of a layer with 10 neurons


def timeit(f, n=1):
    t = time.perf_counter()
    for _ in range(n):
        r = f()
    return r, (time.perf_counter() - t) / n


t1, sec_loops = timeit(lambda: matmul(m1, m2))
t2, sec_torch = timeit(lambda: m1 @ m2, 20)
print(f"\n3 Python loops: {sec_loops * 1000:9.2f} ms")
print(f"PyTorch @:      {sec_torch * 1000:9.4f} ms   ({sec_loops / sec_torch:,.0f}x faster)")
print("same result:", torch.allclose(t1, t2, atol=1e-3))
