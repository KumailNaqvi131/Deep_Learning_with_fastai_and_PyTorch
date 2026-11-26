# Chapter 18: CNN Interpretation with CAM

Which parts of the image made the model decide "cat"? Hooks to grab activations and gradients inside a model, class activation maps (CAM), and Grad-CAM.

## Topics and code

| # | Topic | Book section | File |
|---|-------|--------------|------|
| 1 | `register_forward_hook`; a `Hook` class used as a context manager | CAM and Hooks | `01_hooks.py` |
| 2 | CAM: `einsum('ck,kij->cij', weights, activations)`; the book's approximation and the exact version | CAM and Hooks | `02_class_activation_map.py` |
| 3 | Grad-CAM with a backward hook, on the last layer and an earlier one | Gradient CAM | `03_gradient_cam.py` |

`cat_dog_model.py` trains Chapter 1's cat/dog model once (cached in `outputs/cat_dog.pth`) and draws the heat maps.

## Run

```bash
cd Chapter_18_CNN_Interpretation_with_CAM
python 01_hooks.py
python 02_class_activation_map.py
python 03_gradient_cam.py
```

Look at the pictures in `outputs/`.

## Notes and changes from the book
- fastai's head has two linear layers (after concat pooling), so the book's CAM (using only the last layer's weights) is an approximation. On our model it highlights the wrong area (`outputs/cam_book_approximation.png`). `02` also trains a CAM-compatible head on the frozen body (average pooling + one linear layer, 1 epoch). For that head, CAM is exact: the class score is the average of the map. Its map highlights the cat's head.
- `register_backward_hook` is deprecated in PyTorch. `03` uses `register_full_backward_hook`, which needs an input that requires grad.

## Key takeaways
- Hooks let you read (or change) activations and gradients anywhere in a model without modifying it. Always remove them (use `with`).
- CAM: multiply the final feature maps by the classifier's weights before averaging, to see where each class's evidence is.
- Grad-CAM works on any layer: weight each channel by the average gradient of the class score.
