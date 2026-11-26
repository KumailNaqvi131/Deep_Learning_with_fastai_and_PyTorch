"""
Chapter 15 — Application Architectures Deep Dive
Topic 3: A Siamese network (book section of the same name)

Two pet images in, "same breed?" out. The SAME pretrained encoder processes both images; the
two feature vectors are concatenated and a head predicts same/different. A custom splitter puts
the encoder and the head in different parameter groups, so freeze() and discriminative learning
rates work as for cnn_learner.

    class SiameseModel(Module):
        def __init__(self, encoder, head): self.encoder,self.head = encoder,head
        def forward(self, x1, x2):
            ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
            return self.head(ftrs)
    encoder = create_body(resnet34, cut=-2)
    head = create_head(512*4, 2, ps=0.5)
    learn = Learner(dls, model, loss_func=loss_func, splitter=siamese_splitter, metrics=accuracy)

(The data pipeline is Chapter 11's SiameseTransform, written here as a plain PyTorch Dataset.)

Run:
    python 03_siamese_network.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from common import *  # noqa: F401,F403,E402
from common.vision import _to_tensor  # noqa: E402

set_seed(42)
path = untar_data(URLs.PETS)
files = get_image_files(path / "images")
labeller = RegexLabeller(r"^(.*)_\d+.jpg$")
label_func = lambda f: labeller(f.name)  # noqa: E731
lbl2files = {}
for f in files:
    lbl2files.setdefault(label_func(f), []).append(f)
labels = sorted(lbl2files)
MEAN, STD = torch.tensor(imagenet_stats[0]).view(3, 1, 1), torch.tensor(imagenet_stats[1]).view(3, 1, 1)


class SiameseDataset(Dataset):
    """(image 1, image 2, 1 if same breed else 0). The second image is drawn at random (same breed with
    probability 0.5); for the validation set the draw is made once, so validation is stable."""

    def __init__(self, items, fixed=False):
        self.items = items
        self.pairs = [self._draw(f) for f in items] if fixed else None

    def _draw(self, f):
        same = random.random() < 0.5
        cls = label_func(f)
        if not same:
            cls = random.choice([lbl for lbl in labels if lbl != cls])
        return random.choice(lbl2files[cls]), same

    def _load(self, f):
        img = Resize(224)([PILImage.create(f)], ["image"], train=False)[0]
        return (_to_tensor([img], ["image"])[0].float() / 255 - MEAN) / STD

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f2, same = self.pairs[i] if self.pairs else self._draw(self.items[i])
        return self._load(self.items[i]), self._load(f2), torch.tensor(int(same))


train_idx, valid_idx = RandomSplitter(seed=42)(files)
train_ds, valid_ds = SiameseDataset([files[i] for i in train_idx]), SiameseDataset([files[i] for i in valid_idx], fixed=True)
nw = 4 if sys.platform.startswith("linux") else 0
dls = DataLoaders(DataLoader(train_ds, 32, shuffle=True, num_workers=nw, drop_last=True),
                  DataLoader(valid_ds, 64, num_workers=nw), n_inp=2)


class SiameseModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder, self.head = encoder, head

    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)


encoder = create_body(resnet34, pretrained=True)
head = create_head(512 * 2, 2, ps=0.5)                    # 512 features x 2 images (x2 again for concat pooling)
model = SiameseModel(encoder, head)


def loss_func(out, targ):
    return nn.CrossEntropyLoss()(out, targ.long())


def siamese_splitter(model):
    return [list(model.encoder.parameters()), list(model.head.parameters())]


learn = Learner(dls, model, loss_func=loss_func, splitter=siamese_splitter, metrics=accuracy).to_fp16()
learn.freeze()
learn.fit_one_cycle(4, 3e-3)
learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-6, 1e-4))
