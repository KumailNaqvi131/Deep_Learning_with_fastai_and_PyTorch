"""
Chapter 17 — A Neural Net from the Foundations
Topic 2: Elementwise arithmetic and broadcasting (book sections "Elementwise Arithmetic",
"Broadcasting")

Remove the loops one by one:
  * elementwise: the inner loop over k is (a[i,:] * b[:,j]).sum()
  * broadcasting with a scalar, with a vector (a vector is "expanded" along a new axis without
    copying memory: look at expand_as and its stride of 0), and the general rules: compare shapes
    from the last axis; sizes are compatible when equal, or one of them is 1 (or missing).
  * unsqueeze / None to add an axis: c[:,None] is a column, c[None,:] a row.
  * the matmul with one loop: c[i] = (a[i, :, None] * b).sum(dim=0)

Run:
    python 02_elementwise_and_broadcasting.py
"""

import time

import torch

torch.manual_seed(0)
a, b = torch.tensor([10.0, 6, -4]), torch.tensor([2.0, 8, 7])
print("a + b =", (a + b).tolist(), "| (a < b).float().mean() =", (a < b).float().mean().item())
m = torch.tensor([[1.0, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Frobenius norm (m*m).sum().sqrt() =", (m * m).sum().sqrt().item())

# Broadcasting with a scalar and with a vector
print("\na > 0:", (a > 0).tolist(), "| m / 2 - 1 (scalar broadcast):", (m / 2 - 1).tolist())
c = torch.tensor([10.0, 20, 30])
print("m + c (c added to every row):\n", m + c)
t = c.expand_as(m)
print("c.expand_as(m):\n", t, "\nits storage still has 3 numbers:", t.untyped_storage().size() // 4,
      "| stride:", t.stride(), "(0 = repeat along rows)")
print("c[:, None] (a column):", c[:, None].shape, " c[None, :] (a row):", c[None, :].shape)
print("m + c[:, None] (c added to every column):\n", m + c[:, None])
print("c[None, :] * c[:, None] (outer product via broadcasting):\n", c[None, :] * c[:, None])

# Broadcasting rules: Image  (3d array): 256 x 256 x 3 ;  Scale (1d array): 3  -> result 256 x 256 x 3
img = torch.rand(256, 256, 3)
scale = torch.tensor([0.5, 1.0, 2.0])
print("\n(256,256,3) * (3,) ->", tuple((img * scale).shape))


# Matrix multiplication, loops removed one by one
def matmul_loops(a, b):
    c = torch.zeros(a.shape[0], b.shape[1])
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            for k in range(a.shape[1]):
                c[i, j] += a[i, k] * b[k, j]
    return c


def matmul_elementwise(a, b):
    c = torch.zeros(a.shape[0], b.shape[1])
    for i in range(a.shape[0]):
        for j in range(b.shape[1]):
            c[i, j] = (a[i] * b[:, j]).sum()
    return c


def matmul_broadcast(a, b):
    c = torch.zeros(a.shape[0], b.shape[1])
    for i in range(a.shape[0]):
        c[i] = (a[i, :, None] * b).sum(dim=0)
    return c


m1, m2 = torch.randn(5, 784), torch.randn(784, 10)
for f in (matmul_loops, matmul_elementwise, matmul_broadcast):
    t0 = time.perf_counter()
    r = f(m1, m2)
    print(f"{f.__name__:20s} {(time.perf_counter() - t0) * 1000:9.3f} ms   correct: {torch.allclose(r, m1 @ m2, atol=1e-3)}")
