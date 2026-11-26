"""
Chapter 18 — CNN Interpretation with CAM
Topic 2: Class activation map (book section "CAM and Hooks")

CAM (Zhou et al., 2016): the last convolutional layer gives a 7x7 grid of 512 features; the head
averages each feature over the grid and multiplies by the final layer's weights. So do the
multiplication *before* averaging, at every grid position: we get, for each class, a 7x7 map
of how much each part of the image pushed the prediction towards that class.

    cam_map = torch.einsum('ck,kij->cij', learn.model[1][-1].weight, act)

fastai's head has two linear layers (and concat pooling), so the book's einsum with the last
layer's weights is only an approximation, and it can point at the wrong place. CAM is exact for
a model that ends with average pooling + ONE linear layer: this script also trains such a head
(1 epoch, the body frozen) and computes the exact CAM, to compare.

Run:
    python 02_class_activation_map.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from cat_dog_model import OUT, a_cat_image, get_learner, heatmap_over, prepare  # noqa: E402
from common import CrossEntropyLossFlat, Flatten, Hook, Learner, error_rate  # noqa: E402

learn = get_learner()
img = a_cat_image()
x = prepare(learn, img)

with Hook(learn.model[0], lambda m, i, o: o.detach().clone()) as hook:
    with torch.no_grad():
        output = learn.model.eval()(x)
act = hook.stored[0]                                       # [512, 7, 7]
w = learn.model[1][-1].weight                              # [2 classes, 512]
cam_map = torch.einsum("ck,kij->cij", w, act)
print("activations:", tuple(act.shape), "| last layer weights:", tuple(w.shape), "| cam_map:", tuple(cam_map.shape))
print("prediction:", learn.dls.vocab[int(output.argmax())], torch.softmax(output, -1)[0].tolist())

display = img.resize((224, 224))
heatmap_over(display, cam_map[1], OUT / "cam_book_approximation.png", "book's CAM approximation, 'cat'")

# Exact CAM: average pooling + a single linear layer on top of the same (frozen) body
cam_model = nn.Sequential(learn.model[0], nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(512, 2))
cam_learn = Learner(learn.dls, cam_model, loss_func=CrossEntropyLossFlat(), metrics=error_rate,
                    splitter=lambda m: [list(m[0].parameters()), list(m[3].parameters())])
cam_learn.freeze()
print("\ntraining a CAM-compatible head (avg pool + 1 linear layer) on the frozen body:")
cam_learn.fit_one_cycle(1, 1e-2)
w, b = cam_model[3].weight.detach(), cam_model[3].bias.detach()
with torch.no_grad():
    act = cam_model[0].eval()(x)[0]
cam_map = torch.einsum("ck,kij->cij", w, act) + b[:, None, None]
print("exact CAM: the class scores are the averages of the maps:",
      torch.allclose(cam_map.mean((1, 2)), cam_model.eval()(x)[0].detach(), atol=1e-3))
heatmap_over(display, cam_map[1], OUT / "cam_cat.png", "exact CAM for 'cat' (True)")
heatmap_over(display, cam_map[0], OUT / "cam_dog.png", "exact CAM for 'dog' (False)")
print("Bright areas pushed the prediction towards that class: for 'cat', the cat's head (ears, eyes, whiskers).")
