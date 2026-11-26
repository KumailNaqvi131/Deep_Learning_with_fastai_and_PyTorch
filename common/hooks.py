"""PyTorch hooks (Ch. 13 and 18): look at the activations inside a model while it runs.

    Hook / Hooks       register a forward hook (on one / several modules) and store what it computes
    ActivationStats    a callback recording the mean, std, fraction near zero and histogram of the
                       activations of every layer with parameters, at every training batch
    model_summary      the output shape and number of parameters of every layer (learn.summary())
"""

import torch

from .learner import Callback
from .utils import try_import_matplotlib


class Hook:
    """Calls hook_func(module, input, output) after every forward pass of module; keeps the result
    in self.stored. Use as a context manager so the hook is removed afterwards."""

    def __init__(self, m, hook_func, is_forward=True):
        register = m.register_forward_hook if is_forward else m.register_full_backward_hook
        self.hook_func, self.stored = hook_func, None
        self.hook = register(self._hook)

    def _hook(self, m, i, o):
        self.stored = self.hook_func(m, i, o)

    def remove(self):
        self.hook.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


class Hooks(list):
    def __init__(self, modules, hook_func, is_forward=True):
        super().__init__(Hook(m, hook_func, is_forward) for m in modules)

    def remove(self):
        for h in self:
            h.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


def layers_with_params(model):
    """Leaf modules that have parameters (Conv2d, Linear, BatchNorm, ...), in order."""
    return [m for m in model.modules() if len(list(m.children())) == 0 and len(list(m.parameters(recurse=False))) > 0]


class ActivationStats(Callback):
    """fastai's ActivationStats: learn.activation_stats.plot_layer_stats(idx) and .color_dim(idx)."""
    order = 70

    def __init__(self, with_hist=False, modules=None):
        self.with_hist, self.modules = with_hist, modules

    @property
    def name(self):
        return "activation_stats"

    def before_fit(self):
        self.mods = self.modules or layers_with_params(self.learn.model)
        self.stats = [[] for _ in self.mods]          # per layer: one dict per training batch

        def hook(m, i, o):
            if not self.learn.training:
                return None
            o = o.detach().float()
            res = {"mean": o.mean().item(), "std": o.std().item(), "near_zero": (o <= 0.05).float().mean().item()}
            if self.with_hist:
                res["hist"] = o.histc(40, 0, 10).cpu()
            return res

        self.hooks = Hooks(self.mods, hook)

    def after_batch(self):
        if self.learn.training:
            for s, h in zip(self.stats, self.hooks):
                if h.stored is not None:
                    s.append(h.stored)

    def after_fit(self):
        self.hooks.remove()

    def layer_stats(self, idx):
        s = self.stats[idx]
        return [torch.tensor([d[k] for d in s]) for k in ("mean", "std", "near_zero")]

    def plot_layer_stats(self, idx, fname="layer_stats.png"):
        means, stds, nz = self.layer_stats(idx)
        print(f"layer {idx} ({type(self.mods[idx]).__name__}): mean {means[-1]:.3f}, std {stds[-1]:.3f}, "
              f"near zero {nz[-1]:.0%} at the last batch (std ranged {stds.min():.3f}..{stds.max():.3f})")
        plt = try_import_matplotlib()
        if plt is None:
            return
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        for ax, v, t in zip(axs, (means, stds, nz), ("mean", "std", "% near zero")):
            ax.plot(v)
            ax.set_title(t)
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fname}")

    def color_dim(self, idx, fname="color_dim.png"):
        """Histogram of the activations at each batch, as an image: x = batch, y = activation value
        (0 at the bottom), colour = log(count). A 'crash' shows up as everything falling to zero."""
        plt = try_import_matplotlib()
        if plt is None:
            return
        h = torch.stack([d["hist"] for d in self.stats[idx]]).t().float().log1p()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(h, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xlabel("batch")
        ax.set_ylabel("activation histogram (0 at the bottom)")
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fname}")


def model_summary(model, xb):
    """Like fastai's learn.summary(): output shape and parameters of each layer for one batch."""
    rows = []

    def hook(m, i, o):
        n = sum(p.numel() for p in m.parameters(recurse=False))
        rows.append((type(m).__name__, tuple(o.shape) if isinstance(o, torch.Tensor) else "", n,
                     any(p.requires_grad for p in m.parameters(recurse=False))))

    leaves = [m for m in model.modules() if len(list(m.children())) == 0]
    with Hooks(leaves, hook):
        with torch.no_grad():
            model.eval()
            model(xb)
            model.train()
    print(f"{type(model).__name__} (Input shape: {tuple(xb.shape)})")
    print(f"{'Layer (type)':20s}{'Output Shape':22s}{'Param #':>10s}  Trainable")
    print("=" * 66)
    for name, shape, n, tr in rows:
        print(f"{name:20s}{str(shape):22s}{n:>10,d}  {tr if n else ''}")
    total = sum(r[2] for r in rows)
    print("=" * 66 + f"\nTotal params: {total:,}")
    return rows
