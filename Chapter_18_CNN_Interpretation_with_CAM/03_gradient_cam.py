"""
Chapter 18 — CNN Interpretation with CAM
Topic 3: Gradient CAM (book section "Gradient CAM")

CAM only works on the last layer (it uses the weights of the layer right after it). Grad-CAM
(Selvaraju et al., 2016) works for any layer: the gradient of the class score with respect to a
layer's activations tells how important each channel is; average it over the grid to get one
weight per channel, then weight the activations with it:

    class HookBwd():
        def __init__(self, m): self.hook = m.register_backward_hook(self.hook_func)
        def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
    with HookBwd(learn.model[0]) as hookg:
        with Hook(learn.model[0]) as hook:
            output = learn.model.eval()(x.cuda())
            act = hook.stored
        output[0,cls].backward()
        grad = hookg.stored
    w = grad[0].mean(dim=[1,2], keepdim=True)
    cam_map = (w * act[0]).sum(0)

Earlier layers give finer (higher-resolution) but less "semantic" maps.

Run:
    python 03_gradient_cam.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_dog_model import OUT, a_cat_image, get_learner, heatmap_over, prepare  # noqa: E402


class Hook:
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)

    def hook_func(self, m, i, o):
        self.stored = o.detach().clone()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.hook.remove()


class HookBwd:
    def __init__(self, m):
        self.hook = m.register_full_backward_hook(self.hook_func)   # (register_backward_hook in the book)

    def hook_func(self, m, gi, go):
        self.stored = go[0].detach().clone()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.hook.remove()


learn = get_learner()
img = a_cat_image()
x = prepare(learn, img).requires_grad_()            # (a full backward hook needs an input that requires grad)
display = img.resize((224, 224))
cls = 1                                             # True = cat

for name, layer in [("last layer (learn.model[0])", learn.model[0]), ("an earlier layer (learn.model[0][-2])", learn.model[0][-2])]:
    with HookBwd(layer) as hookg:
        with Hook(layer) as hook:
            output = learn.model.eval()(x)
            act = hook.stored
        output[0, cls].backward()
        grad = hookg.stored
    w = grad[0].mean(dim=[1, 2], keepdim=True)      # one importance weight per channel
    cam_map = (w * act[0]).sum(0)
    print(f"{name}: activations {tuple(act.shape)}, gradients {tuple(grad.shape)}, map {tuple(cam_map.shape)}")
    heatmap_over(display, cam_map, OUT / f"gradcam_{'last' if layer is learn.model[0] else 'earlier'}.png", f"Grad-CAM, {name}")
