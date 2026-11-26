"""Model building blocks used by several chapters: the fastai CNN head, pretrained bodies,
xResNet (Ch. 7 and 14) and a small U-Net with a ResNet encoder (Ch. 1 and 15)."""

import warnings

import torch
import torch.nn.functional as F
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class AdaptiveConcatPool2d(nn.Module):
    """Average pooling AND max pooling, concatenated: twice the features of plain avg pooling (Ch. 15)."""

    def __init__(self, size=1):
        super().__init__()
        self.ap, self.mp = nn.AdaptiveAvgPool2d(size), nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Embedding(nn.Embedding):
    """fastai's Embedding: an nn.Embedding initialised with small values (truncated normal, std 0.01)
    instead of PyTorch's N(0, 1), so dot products of embeddings start near 0."""

    def __init__(self, ni, nf, std=0.01):
        super().__init__(ni, nf)
        nn.init.trunc_normal_(self.weight, std=std, a=-2 * std, b=2 * std)


def sigmoid_range(x, lo, hi):
    """Squash activations into (lo, hi) (Ch. 6 and 8)."""
    return torch.sigmoid(x) * (hi - lo) + lo


class SigmoidRange(nn.Module):
    def __init__(self, lo, hi):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x):
        return sigmoid_range(x, self.lo, self.hi)


# --------------------------------------------------------------------------------------------------
# Pretrained bodies + the fastai head
# --------------------------------------------------------------------------------------------------
def load_arch(arch, pretrained=True):
    """Instantiate a torchvision model function (resnet34, ...), with ImageNet weights if asked.
    If the weights can't be downloaded (offline), fall back to random weights with a warning."""
    if not pretrained:
        try:
            return arch(weights=None)
        except TypeError:
            return arch()
    try:
        return arch(weights="DEFAULT")
    except TypeError:                                  # our own architectures (xresnet) have no weights
        return arch()
    except Exception as e:                             # no internet, ...
        warnings.warn(f"Could not load pretrained weights ({type(e).__name__}: {e}). Using random weights.")
        return arch(weights=None)


def create_body(arch, pretrained=True, cut=None):
    """Everything in the model up to (not including) the final pooling + classifier."""
    model = load_arch(arch, pretrained) if not isinstance(arch, nn.Module) else arch
    children = list(model.children())
    if cut is None:
        cut = next(i for i, c in enumerate(children) if isinstance(c, (nn.AdaptiveAvgPool2d, AdaptiveConcatPool2d)))
    return nn.Sequential(*children[:cut])


def num_features_model(body, size=64):
    """Number of channels the body outputs (found by passing a dummy image through it)."""
    p = next(body.parameters())
    with torch.no_grad():
        was = body.training
        body.eval()
        out = body(torch.zeros(2, 3, size, size, device=p.device))
        body.train(was)
    return out.shape[1]


def create_head(nf, n_out, lin_ftrs=None, ps=0.5, concat_pool=True, first_bn=True, bn_final=False, y_range=None):
    """fastai's classifier head: pool -> flatten -> [BatchNorm, Dropout, Linear, ReLU] x2 (Ch. 15)."""
    if concat_pool:
        nf *= 2
    lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf] + list(lin_ftrs) + [n_out]
    ps = [ps] if not isinstance(ps, (list, tuple)) else list(ps)
    if len(ps) == 1:
        ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
    acts = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    layers = [AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1), Flatten()]
    for i, (ni, no, p, act) in enumerate(zip(lin_ftrs[:-1], lin_ftrs[1:], ps, acts)):
        if i > 0 or first_bn:
            layers.append(nn.BatchNorm1d(ni))
        layers += [nn.Dropout(p), nn.Linear(ni, no, bias=not (i > 0 or first_bn))]
        if act is not None:
            layers.append(act)
    if bn_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    if y_range is not None:
        layers.append(SigmoidRange(*y_range))
    return nn.Sequential(*layers)


def init_head(m):
    for layer in m.modules():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


def create_cnn_model(arch, n_out, pretrained=True, cut=None, concat_pool=True, lin_ftrs=None, ps=0.5, y_range=None):
    body = create_body(arch, pretrained, cut)
    nf = num_features_model(body)
    head = create_head(nf, n_out, lin_ftrs=lin_ftrs, ps=ps, concat_pool=concat_pool, y_range=y_range)
    init_head(head)
    return nn.Sequential(body, head)


def cnn_splitter(m):
    """Three parameter groups (early body, late body, head) for freezing and discriminative lrs."""
    if hasattr(m, "encoder"):                              # U-Net
        body, rest = m.encoder, [m.decoder]
    else:
        body, rest = m[0], list(m)[1:]
    children = list(body.children())
    k = max(1, len(children) * 3 // 4)
    params = lambda mods: [p for mod in mods for p in mod.parameters()]  # noqa: E731
    return [params(children[:k]), params(children[k:]), params(rest)]


# --------------------------------------------------------------------------------------------------
# xResNet: ResNet with the "bag of tricks" tweaks (Ch. 7 and 14)
# --------------------------------------------------------------------------------------------------
def ConvLayer(ni, nf, ks=3, stride=1, act=True, zero_bn=False, ndim=2):
    """Conv -> BatchNorm -> (ReLU). zero_bn initialises the BatchNorm weight to 0 (used for the last
    layer of a res block, so that each block starts out as the identity)."""
    conv = nn.Conv2d(ni, nf, ks, stride=stride, padding=ks // 2, bias=False)
    nn.init.kaiming_normal_(conv.weight)
    bn = nn.BatchNorm2d(nf)
    nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)
    layers = [conv, bn] + ([nn.ReLU(inplace=True)] if act else [])
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    """x -> relu(convs(x) + id(x)). expansion=4 gives the bottleneck version (1x1, 3x3, 1x1)."""

    def __init__(self, expansion, ni, nf, stride=1):
        super().__init__()
        nh, ni, nf = nf, ni * expansion, nf * expansion
        if expansion == 1:
            self.convs = nn.Sequential(ConvLayer(ni, nh, stride=stride), ConvLayer(nh, nf, act=False, zero_bn=True))
        else:
            self.convs = nn.Sequential(ConvLayer(ni, nh, 1), ConvLayer(nh, nh, stride=stride),
                                       ConvLayer(nh, nf, 1, act=False, zero_bn=True))
        self.idconv = nn.Identity() if ni == nf else ConvLayer(ni, nf, 1, act=False)
        self.pool = nn.Identity() if stride == 1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))


class XResNet(nn.Sequential):
    def __init__(self, expansion, layers, n_out=1000, c_in=3, p=0.0, widths=(64, 128, 256, 512)):
        stem = [ConvLayer(c_in, 32, stride=2), ConvLayer(32, 32), ConvLayer(32, 64)]
        sizes = [64 // expansion] + list(widths)
        blocks = []
        for i, n in enumerate(layers):
            stride = 1 if i == 0 else 2
            blocks.append(nn.Sequential(*[ResBlock(expansion, sizes[i] if j == 0 else sizes[i + 1],
                                                   sizes[i + 1], stride if j == 0 else 1) for j in range(n)]))
        super().__init__(*stem, nn.MaxPool2d(3, stride=2, padding=1), *blocks,
                         nn.AdaptiveAvgPool2d(1), Flatten(), nn.Dropout(p), nn.Linear(sizes[-1] * expansion, n_out))


def xresnet18(n_out=1000, c_in=3, p=0.0, **kw):
    return XResNet(1, [2, 2, 2, 2], n_out, c_in, p)


def xresnet34(n_out=1000, c_in=3, p=0.0, **kw):
    return XResNet(1, [3, 4, 6, 3], n_out, c_in, p)


def xresnet50(n_out=1000, c_in=3, p=0.0, **kw):
    return XResNet(4, [3, 4, 6, 3], n_out, c_in, p)


# --------------------------------------------------------------------------------------------------
# U-Net with a ResNet encoder (a simplified version of fastai's DynamicUnet)
# --------------------------------------------------------------------------------------------------
class UnetBlock(nn.Module):
    """Upsample the deeper features x2, concatenate the skip connection from the encoder, then 2 convs."""

    def __init__(self, up_in, skip_in, out):
        super().__init__()
        self.up = nn.Sequential(nn.Conv2d(up_in, up_in // 2, 1), nn.ReLU(inplace=True))
        self.bn = nn.BatchNorm2d(skip_in)
        self.convs = nn.Sequential(ConvLayer(up_in // 2 + skip_in, out), ConvLayer(out, out))

    def forward(self, x, skip):
        x = F.interpolate(self.up(x), size=skip.shape[-2:], mode="nearest")
        return self.convs(torch.cat([x, self.bn(skip)], dim=1))


class SimpleUnet(nn.Module):
    def __init__(self, arch, n_out, pretrained=True):
        super().__init__()
        self.encoder = create_body(arch, pretrained)        # conv1 bn1 relu maxpool layer1..layer4
        x = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            self.encoder.eval()
            feats = self._features(x)
            self.encoder.train()
        chans = [f.shape[1] for f in feats]                  # stride 2, 4, 8, 16, 32
        mid = chans[-1]
        self.middle = nn.Sequential(ConvLayer(mid, mid * 2), ConvLayer(mid * 2, mid))
        ups, c = [], mid
        for skip_c in reversed(chans[:-1]):
            out = max(skip_c, 32)
            ups.append(UnetBlock(c, skip_c, out))
            c = out
        self.ups = nn.ModuleList(ups)
        self.final = nn.Sequential(ConvLayer(c + 3, 32), nn.Conv2d(32, n_out, 1))
        self.decoder = nn.ModuleList([self.middle, self.ups, self.final])

    def _features(self, x):
        feats = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in (2, 4, 5, 6, 7):                         # after relu, layer1, layer2, layer3, layer4
                feats.append(x)
        return feats

    def forward(self, x):
        inp = x
        feats = self._features(x)
        x = self.middle(feats[-1])
        for up, skip in zip(self.ups, reversed(feats[:-1])):
            x = up(x, skip)
        x = F.interpolate(x, size=inp.shape[-2:], mode="bilinear", align_corners=False)
        return self.final(torch.cat([x, inp], dim=1))
