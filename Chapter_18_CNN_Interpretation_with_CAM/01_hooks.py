"""
Chapter 18 — CNN Interpretation with CAM
Topic 1: Hooks (book section "CAM and Hooks")

A hook is a function PyTorch calls every time a module runs: register_forward_hook(f) calls
f(module, input, output) after the module's forward. That's how we can grab activations from
inside a model without changing its code. Always remove a hook when done (or it keeps running,
and keeps memory alive): a context manager does that for us.

    class Hook():
        def __init__(self, m): self.hook = m.register_forward_hook(self.hook_func)
        def hook_func(self, m, i, o): self.stored = o.detach().clone()
        def __enter__(self, *args): return self
        def __exit__(self, *args): self.hook.remove()

Run:
    python 01_hooks.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cat_dog_model import a_cat_image, get_learner, prepare  # noqa: E402

learn = get_learner()
x = prepare(learn, a_cat_image())


class HookV1:
    def hook_func(self, m, i, o):
        self.stored = o.detach().clone()


hook_output = HookV1()
hook = learn.model[0].register_forward_hook(hook_output.hook_func)
with torch.no_grad():
    output = learn.model.eval()(x)
act = hook_output.stored[0]
hook.remove()
print("input:", tuple(x.shape), "-> activations of the body (learn.model[0]):", tuple(act.shape), "-> output:", tuple(output.shape))
print("predicted probabilities (False=dog, True=cat):", torch.softmax(output, dim=-1)[0].tolist(), "| vocab:", learn.dls.vocab)


class Hook:
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)

    def hook_func(self, m, i, o):
        self.stored = o.detach().clone()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.hook.remove()


with Hook(learn.model[0]) as hook:
    with torch.no_grad():
        learn.model.eval()(x)
    print("\nwith a context manager: the hook stored", tuple(hook.stored.shape),
          "and is removed when the `with` block ends")
print("hooks still registered on the body:", len(learn.model[0]._forward_hooks))
