"""
Chapter 1 — Your Deep Learning Journey
Topic 4: Image recognizers can tackle non-image tasks (book section of the same name)

Lots of data can be turned into a picture: the book shows sounds converted to spectrograms,
time series, mouse movements and malware binaries. A pretrained image model, fine-tuned on
those pictures, often works remarkably well.

Here we synthesise 1-second sounds of three kinds (a whistle with vibrato, a rising chirp,
and knocking), convert each one into a spectrogram (time on the x-axis, frequency on the y-axis)
with torch.stft, save the spectrograms as PNG files, and fine-tune a pretrained ResNet on them,
exactly as if they were photos.

Run:
    python 04_non_image_tasks_as_images.py
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from common import *  # noqa: F401,F403,E402

OUT = outputs_dir(__file__)
set_seed(42)
SR = 8000                                          # samples per second
SOUNDS = DATA_ROOT / "sound_spectrograms"


def make_sound(kind, g):
    t = torch.arange(SR) / SR
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(1, generator=g).item()  # noqa: E731
    if kind == "whistle":                           # a steady tone with vibrato
        f0 = r(600, 2500)
        wave = torch.sin(2 * math.pi * (f0 * t + 8 * torch.sin(2 * math.pi * r(3, 7) * t)))
    elif kind == "chirp":                           # frequency rising over time
        f0, f1 = r(200, 800), r(1500, 3500)
        wave = torch.sin(2 * math.pi * (f0 * t + (f1 - f0) / 2 * t ** 2))
    else:                                           # knock: short bursts of noise
        wave = torch.zeros(SR)
        for _ in range(int(r(2, 6))):
            s = int(r(0, SR - 400))
            wave[s:s + 400] += torch.randn(400, generator=g) * torch.exp(-torch.arange(400) / 80.0)
    return wave * r(0.3, 1.0) + 0.05 * torch.randn(SR, generator=g)   # + background noise


def spectrogram(wave):
    spec = torch.stft(wave, n_fft=256, hop_length=64, window=torch.hann_window(256), return_complex=True).abs()
    spec = torch.log1p(spec * 10)                   # log scale, like our ears
    spec = (spec / spec.max() * 255).flip(0)        # low frequencies at the bottom
    return Image.fromarray(spec.byte().numpy()).convert("RGB")


if not (SOUNDS / ".complete").exists():
    print("Synthesising sounds and saving their spectrograms ...")
    g = torch.Generator().manual_seed(0)
    for kind in ("whistle", "chirp", "knock"):
        (SOUNDS / kind).mkdir(parents=True, exist_ok=True)
        for i in range(120):
            spectrogram(make_sound(kind, g)).save(SOUNDS / kind / f"{kind}_{i}.png")
    (SOUNDS / ".complete").write_text("ok")

# From here on it is ordinary image classification
dls = DataBlock(blocks=(ImageBlock, CategoryBlock), get_items=get_image_files, get_y=parent_label,
                splitter=RandomSplitter(valid_pct=0.2, seed=42),
                item_tfms=Resize(128, method="squish")).dataloaders(SOUNDS, bs=32)
dls.show_batch(max_n=6, fname=OUT / "spectrograms.png")
print("Look at outputs/spectrograms.png: whistles are wavy horizontal lines, chirps rise, knocks are vertical bars.")

learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(2)
