"""
Chapter 17 — A Neural Net from the Foundations
Topic 3: Einstein summation (book section of the same name)

torch.einsum describes products and sums with index letters:
    'ik,kj->ij'      matrix multiplication (k is repeated and absent from the output: summed over)
    'ij->ji'         transpose
    'bi,ij,bj->b'    for each b, sum over i and j of a[b,i]*b[i,j]*c[b,j]
    'bik,bkj->bij'   batch matrix multiplication
Rules: repeated letters on the left are multiplied; letters missing on the right are summed.
Compact, readable, and fast.

Run:
    python 03_einstein_summation.py
"""

import time

import torch

torch.manual_seed(0)


def matmul(a, b):
    return torch.einsum("ik,kj->ij", a, b)


m1, m2 = torch.randn(5, 784), torch.randn(784, 10)
t0 = time.perf_counter()
for _ in range(20):
    r = matmul(m1, m2)
print(f"einsum matmul: {(time.perf_counter() - t0) / 20 * 1000:.4f} ms, correct: {torch.allclose(r, m1 @ m2, atol=1e-4)}")

a = torch.randn(3, 4)
print("einsum('ij->ji', a) == a.T:", torch.equal(torch.einsum("ij->ji", a), a.T))
x, w, y = torch.randn(6, 3), torch.randn(3, 3), torch.randn(6, 3)
loop = torch.stack([sum(x[b, i] * w[i, j] * y[b, j] for i in range(3) for j in range(3)) for b in range(6)])
print("einsum('bi,ij,bj->b') == loops:", torch.allclose(torch.einsum("bi,ij,bj->b", x, w, y), loop, atol=1e-5))
p, q = torch.randn(8, 2, 3), torch.randn(8, 3, 4)
print("einsum('bik,bkj->bij') == p @ q (batched):", torch.allclose(torch.einsum("bik,bkj->bij", p, q), p @ q, atol=1e-5))
