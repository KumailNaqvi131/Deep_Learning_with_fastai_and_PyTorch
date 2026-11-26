"""Offline look-alikes of the book's image datasets, drawn with Pillow.

Each generator writes files in the *same folder layout and naming scheme* as the real
dataset, so the chapter code is identical for real and synthetic data:

    PETS            oxford-iiit-pet/images/<Breed>_<n>.jpg   (cat breeds Capitalised, dogs lowercase)
    BEARS           bears/<grizzly|black|teddy>/<n>.jpg        (+ a few broken files to find)
    MNIST_SAMPLE    mnist_sample/{train,valid}/{3,7}/<n>.png
    MNIST           mnist_png/{training,testing}/<digit>/<n>.png
    PASCAL_2007     pascal_2007/train/<n>.jpg + train.csv (fname, labels, is_valid)
    BIWI_HEAD_POSE  biwi_head_pose/<person>/frame_<n>_rgb.jpg + _pose.txt, rgb.cal
    IMAGENETTE      imagenette2-160/{train,val}/<wordnet id>/<n>.JPEG
    CAMVID_TINY     camvid_tiny/{images,labels}/<n>.png, <n>_P.png + codes.txt

The pictures are cartoons, but they keep what matters for learning: many classes that
differ in shape, colour and texture, random poses, sizes, backgrounds, clutter and noise,
some classes that are genuinely hard to tell apart, and images of different sizes.
"""

import math
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --------------------------------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------------------------------
SCENES = [(196, 214, 230), (120, 160, 90), (200, 190, 170), (140, 140, 140), (70, 70, 85),
          (90, 130, 190), (170, 140, 100), (225, 225, 215), (110, 90, 70), (60, 100, 60)]


def jitter(rng, color, amount=25):
    return tuple(int(np.clip(c + rng.integers(-amount, amount + 1), 0, 255)) for c in color)


def background(rng, w, h, base=None):
    """Two-tone background with random blobs and stripes (clutter the model must ignore)."""
    top = jitter(rng, base if base is not None and rng.random() < 0.5
                 else SCENES[rng.integers(len(SCENES))])
    img = Image.new("RGB", (w, h), top)
    d = ImageDraw.Draw(img)
    if rng.random() < 0.6:                                     # a floor / horizon
        hy = int(h * rng.uniform(0.45, 0.8))
        d.rectangle([0, hy, w, h], fill=jitter(rng, SCENES[rng.integers(len(SCENES))]))
    for _ in range(rng.integers(2, 7)):                        # clutter
        x, y, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.03, 0.12) * min(w, h)
        col = tuple(int(v) for v in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            d.ellipse([x - r, y - r * 0.7, x + r, y + r * 0.7], fill=col)
        else:
            d.rectangle([x - r, y - r / 3, x + r, y + r / 3], fill=col)
    return img


def finish(img, rng, noise=8.0, blur=True):
    """Lighting change, a little blur and pixel noise, so no two images are identical."""
    a = np.asarray(img).astype(np.float32)
    a = a * rng.uniform(0.75, 1.2) + rng.uniform(-20, 20)       # brightness / contrast
    a += rng.normal(0, noise, a.shape)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if blur and rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    return img


def ellipse(d, cx, cy, rx, ry, fill, outline=None, width=1):
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill, outline=outline, width=width)


def poly(d, pts, fill):
    d.polygon([(float(x), float(y)) for x, y in pts], fill=fill)


def rotated(pts, cx, cy, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in pts]


def darker(col, f=0.55):
    return tuple(int(v * f) for v in col)


# --------------------------------------------------------------------------------------------------
# Animals: pets and bears
# --------------------------------------------------------------------------------------------------
FUR = {"cream": (232, 215, 180), "ginger": (215, 130, 55), "brown": (130, 85, 50),
       "black": (35, 32, 30), "grey": (135, 145, 160), "white": (245, 245, 240),
       "tan": (195, 160, 110), "pink": (230, 180, 170), "golden": (220, 175, 80),
       "dark": (80, 60, 45)}
EYES = {"blue": (70, 140, 230), "green": (90, 180, 80), "yellow": (230, 200, 40),
        "brown": (110, 70, 30), "orange": (240, 140, 30)}

# breed: (fur, pattern, eyes, ears, snout, fluffy)
#   pattern: plain | spots | stripes | mask (dark face/ears) | patches | saddle
#   ears:    pointy | big_pointy | floppy | small_up | round
#   snout:   0 (flat cat face) .. 1.0 (long dog muzzle)
CAT_BREEDS = {
    "Abyssinian": ("tan", "plain", "green", "big_pointy", 0.0, False),
    "Bengal": ("ginger", "spots", "green", "pointy", 0.0, False),
    "Birman": ("cream", "mask", "blue", "pointy", 0.0, True),
    "Bombay": ("black", "plain", "yellow", "pointy", 0.0, False),
    "British_Shorthair": ("grey", "plain", "orange", "pointy", 0.0, False),
    "Egyptian_Mau": ("grey", "spots", "green", "pointy", 0.0, False),
    "Maine_Coon": ("brown", "stripes", "yellow", "big_pointy", 0.0, True),
    "Persian": ("white", "plain", "orange", "small_up", 0.0, True),
    "Ragdoll": ("cream", "mask", "blue", "pointy", 0.0, False),
    "Russian_Blue": ("grey", "plain", "green", "pointy", 0.0, False),
    "Siamese": ("cream", "mask", "blue", "big_pointy", 0.0, False),
    "Sphynx": ("pink", "plain", "green", "big_pointy", 0.0, False),
}
DOG_BREEDS = {
    "american_bulldog": ("white", "patches", "brown", "floppy", 0.45, False),
    "american_pit_bull_terrier": ("grey", "plain", "brown", "small_up", 0.5, False),
    "basset_hound": ("tan", "saddle", "brown", "floppy", 0.8, False),
    "beagle": ("tan", "saddle", "brown", "floppy", 0.6, False),
    "boxer": ("ginger", "mask", "brown", "small_up", 0.35, False),
    "chihuahua": ("tan", "plain", "brown", "big_pointy", 0.3, False),
    "english_cocker_spaniel": ("golden", "plain", "brown", "floppy", 0.6, True),
    "english_setter": ("white", "spots", "brown", "floppy", 0.8, False),
    "german_shorthaired": ("brown", "spots", "brown", "floppy", 0.85, False),
    "great_pyrenees": ("white", "plain", "brown", "floppy", 0.6, True),
    "havanese": ("cream", "plain", "brown", "floppy", 0.3, True),
    "japanese_chin": ("white", "patches", "brown", "floppy", 0.1, True),
    "keeshond": ("grey", "mask", "brown", "small_up", 0.5, True),
    "leonberger": ("golden", "mask", "brown", "floppy", 0.6, True),
    "miniature_pinscher": ("black", "saddle", "brown", "big_pointy", 0.6, False),
    "newfoundland": ("black", "plain", "brown", "floppy", 0.6, True),
    "pomeranian": ("ginger", "plain", "brown", "small_up", 0.3, True),
    "pug": ("tan", "mask", "brown", "floppy", 0.1, False),
    "saint_bernard": ("white", "patches", "brown", "floppy", 0.6, True),
    "samoyed": ("white", "plain", "brown", "small_up", 0.55, True),
    "scottish_terrier": ("black", "plain", "brown", "small_up", 0.75, True),
    "shiba_inu": ("ginger", "plain", "brown", "small_up", 0.55, False),
    "staffordshire_bull_terrier": ("brown", "patches", "brown", "small_up", 0.45, False),
    "wheaten_terrier": ("golden", "plain", "brown", "floppy", 0.55, True),
    "yorkshire_terrier": ("tan", "saddle", "brown", "small_up", 0.4, True),
}
PET_BREEDS = {**CAT_BREEDS, **DOG_BREEDS}


def draw_animal(img, rng, fur, pattern, eyes, ears, snout, fluffy, cx, cy, s, angle=0.0, extra=None):
    """Draw an animal head + body centred at (cx, cy) with head radius s."""
    d = ImageDraw.Draw(img)
    fur_c = jitter(rng, FUR[fur], 18)
    dark = darker(fur_c, 0.45) if fur != "black" else (90, 80, 70)
    ear_c = dark if pattern == "mask" else fur_c
    # body (behind the head)
    ellipse(d, cx + s * 0.2, cy + s * 1.35, s * 1.15, s * 0.9, fur_c)
    if pattern == "saddle":
        ellipse(d, cx + s * 0.3, cy + s * 1.2, s * 0.8, s * 0.5, dark)
    # ears
    if ears in ("pointy", "big_pointy", "small_up"):
        k = {"pointy": 1.0, "big_pointy": 1.45, "small_up": 0.6}[ears]
        for side in (-1, 1):
            pts = [(cx + side * s * 0.25, cy - s * 0.7), (cx + side * s * 0.9, cy - s * 0.35),
                   (cx + side * s * 0.75, cy - s * (0.75 + 0.75 * k))]
            poly(d, rotated(pts, cx, cy, angle), ear_c)
            if ears != "small_up":
                mx, my = sum(p[0] for p in pts) / 3, sum(p[1] for p in pts) / 3
                inner = [(mx + (x - mx) * 0.5, my + (y - my) * 0.5) for x, y in pts]   # pink inside
                poly(d, rotated(inner, cx, cy, angle), (230, 170, 170))
    elif ears == "floppy":
        for side in (-1, 1):
            ex, ey = cx + side * s * 0.95, cy + s * 0.15
            ex, ey = rotated([(ex, ey)], cx, cy, angle)[0]
            ellipse(d, ex, ey, s * 0.3, s * 0.65, dark if pattern != "plain" else darker(fur_c, 0.8))
    # head (fluffy breeds get a ruff of small circles)
    if fluffy:
        for t in np.linspace(0, 2 * np.pi, 18, endpoint=False):
            ellipse(d, cx + math.cos(t) * s * 0.95, cy + math.sin(t) * s * 0.9, s * 0.28, s * 0.28, fur_c)
    ellipse(d, cx, cy, s, s * 0.92, fur_c)
    # coat pattern on the face
    if pattern == "mask":
        ellipse(d, cx, cy + s * 0.2, s * 0.62, s * 0.6, dark)
    elif pattern == "spots":
        for _ in range(9):
            a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0.2, 0.8) * s
            ellipse(d, cx + math.cos(a) * r, cy + math.sin(a) * r * 0.9, s * 0.1, s * 0.1, dark)
    elif pattern == "stripes":
        for k in (-0.45, -0.15, 0.15, 0.45):
            d.line([(cx + k * s, cy - s * 0.85), (cx + k * s * 1.2, cy - s * 0.35)], fill=dark,
                   width=max(2, int(s * 0.09)))
    elif pattern == "patches":
        side = rng.choice([-1, 1])
        ellipse(d, cx + side * s * 0.45, cy - s * 0.25, s * 0.42, s * 0.48, dark)
    # snout (dogs) and eyes
    if snout > 0:
        ellipse(d, cx, cy + s * (0.3 + 0.2 * snout), s * (0.35 + 0.15 * snout), s * (0.25 + 0.35 * snout),
                jitter(rng, (235, 225, 210), 10) if fur in ("black", "dark", "brown") else darker(fur_c, 0.92))
    for side in (-1, 1):
        ex, ey = cx + side * s * 0.38, cy - s * 0.15
        ellipse(d, ex, ey, s * 0.16, s * 0.13, (250, 250, 245))
        ellipse(d, ex, ey, s * 0.1, s * 0.11, EYES[eyes])
        ellipse(d, ex, ey, s * 0.035, s * (0.09 if snout == 0 else 0.05), (10, 10, 10))
    nose_y = cy + s * (0.15 + 0.4 * snout)
    ellipse(d, cx, nose_y, s * 0.09 + s * 0.05 * snout, s * 0.06 + s * 0.03 * snout,
            (220, 120, 130) if snout == 0 else (25, 20, 20))
    if snout == 0:                                                     # whiskers
        for side in (-1, 1):
            for k in (-0.08, 0.0, 0.08):
                d.line([(cx + side * s * 0.2, nose_y + s * k), (cx + side * s * 0.95, nose_y + s * (k * 2.2))],
                       fill=(240, 240, 240) if fur in ("black", "grey") else (40, 40, 40), width=1)
    if extra == "bow":                                                 # teddy bears wear a bow tie
        by = cy + s * 0.95
        poly(d, [(cx, by), (cx - s * 0.45, by - s * 0.25), (cx - s * 0.45, by + s * 0.25)], (200, 30, 50))
        poly(d, [(cx, by), (cx + s * 0.45, by - s * 0.25), (cx + s * 0.45, by + s * 0.25)], (200, 30, 50))


def pet_image(breed, rng, size=None):
    w, h = size or (int(rng.integers(260, 500)), int(rng.integers(260, 500)))
    img = background(rng, w, h)
    s = min(w, h) * rng.uniform(0.14, 0.24)
    cx, cy = rng.uniform(s * 1.3, w - s * 1.3), rng.uniform(s * 1.6, h - s * 2.0)
    draw_animal(img, rng, *PET_BREEDS[breed], cx, cy, s, angle=rng.uniform(-0.25, 0.25))
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return finish(img, rng)


def make_pets(root, n_per_breed=100, seed=0):
    """oxford-iiit-pet/images/<Breed>_<n>.jpg. Cat breeds start with a capital letter."""
    rng = np.random.default_rng(seed)
    out = root / "images"
    out.mkdir(parents=True, exist_ok=True)
    for breed in PET_BREEDS:
        for i in range(1, n_per_breed + 1):
            pet_image(breed, rng).save(out / f"{breed}_{i}.jpg", quality=90)


BEARS = {"grizzly": ("brown", "plain", "brown", "round", 0.55, True, None),
         "black": ("black", "plain", "brown", "round", 0.6, False, None),
         "teddy": ("golden", "plain", "brown", "round", 0.25, False, "bow")}


def draw_bear(img, rng, kind, cx, cy, s):
    fur, pattern, eyes, _, snout, fluffy, extra = BEARS[kind]
    d = ImageDraw.Draw(img)
    fur_c = jitter(rng, FUR[fur], 15)
    for side in (-1, 1):                                       # round ears
        ellipse(d, cx + side * s * 0.75, cy - s * 0.75, s * 0.32, s * 0.32, fur_c)
        ellipse(d, cx + side * s * 0.75, cy - s * 0.75, s * 0.16, s * 0.16, darker(fur_c, 0.7))
    if kind == "grizzly":                                      # the shoulder hump
        ellipse(d, cx + s * 0.9, cy + s * 0.5, s * 1.0, s * 0.8, darker(fur_c, 0.9))
    draw_animal(img, rng, fur, pattern, eyes, "none", snout, fluffy, cx, cy, s, extra=extra)
    if kind == "black":                                        # tan muzzle
        ellipse(d, cx, cy + s * 0.42, s * 0.36, s * 0.3, (190, 150, 105))
        ellipse(d, cx, cy + s * 0.35, s * 0.1, s * 0.07, (20, 20, 20))
    if kind == "teddy":                                        # stitching
        d.line([(cx, cy + s * 0.5), (cx, cy + s * 0.8)], fill=(90, 60, 30), width=2)


def make_bears(root, n_per_class=150, seed=1):
    """bears/<kind>/<n>.jpg, the dataset you would download with an image search (Ch. 2)."""
    rng = np.random.default_rng(seed)
    for kind in BEARS:
        out = root / kind
        out.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            w, h = int(rng.integers(300, 640)), int(rng.integers(260, 480))
            img = background(rng, w, h, base=(60, 100, 60) if kind != "teddy" else (225, 225, 215))
            s = min(w, h) * rng.uniform(0.15, 0.25)
            draw_bear(img, rng, kind, rng.uniform(s * 1.5, w - s * 2), rng.uniform(s * 1.4, h - s * 2.3), s)
            finish(img, rng).save(out / f"{i:08d}.jpg", quality=88)
    # an image search also returns wrong pictures: a few teddies filed as grizzlies, a grizzly filed as
    # a black bear, and two pictures with no bear at all (Chapter 2 finds them with plot_top_losses)
    def extra(kind_drawn, folder, i):
        w, h = 480, 360
        img = background(rng, w, h)
        if kind_drawn == "car":
            _icon(ImageDraw.Draw(img), "car", w / 2, h / 2, 90, rng)
        else:
            draw_bear(img, rng, kind_drawn, w / 2, h / 2 - 20, 70)
        finish(img, rng).save(root / folder / f"{n_per_class + i:08d}.jpg", quality=88)
    for i in range(4):
        extra("teddy", "grizzly", i)
    extra("grizzly", "black", 0)
    extra("car", "black", 1)
    extra("car", "teddy", 0)
    # ... and a few files that aren't images at all: Chapter 2 finds and removes them
    (root / "grizzly" / "broken_download.jpg").write_bytes(b"<html>404 Not Found</html>")
    (root / "black" / "truncated.jpg").write_bytes((root / "black" / "00000000.jpg").read_bytes()[:300])


# --------------------------------------------------------------------------------------------------
# Hand-written-looking digits (MNIST)
# --------------------------------------------------------------------------------------------------
_FONTS = {}


def _font(size):
    if size not in _FONTS:
        try:
            _FONTS[size] = ImageFont.load_default(size=size)       # Pillow >= 10.1 (TrueType)
        except TypeError:
            _FONTS[size] = ImageFont.load_default()                 # very old Pillow: bitmap font
    return _FONTS[size]


def digit_image(digit, rng):
    """A 28x28 grey-scale digit, white on black, roughly centred like MNIST."""
    big = Image.new("L", (84, 84), 0)
    d = ImageDraw.Draw(big)
    font = _font(int(rng.integers(44, 60)))
    d.text((42 + rng.uniform(-4, 4), 42 + rng.uniform(-4, 4)), str(digit), fill=255, font=font,
           anchor="mm", stroke_width=int(rng.integers(1, 6)), stroke_fill=255)
    # random affine: rotation, shear, aspect -> every digit looks a bit hand-written
    a, sh, sx = rng.uniform(-0.3, 0.3), rng.uniform(-0.35, 0.35), rng.uniform(0.8, 1.2)
    m = np.array([[math.cos(a) * sx, -math.sin(a) + sh, 0], [math.sin(a), math.cos(a), 0]])
    c = np.array([42, 42])
    m[:, 2] = c - m[:, :2] @ c
    big = big.transform((84, 84), Image.AFFINE, tuple(m.flatten()), resample=Image.BILINEAR)
    if rng.random() < 0.4:                                  # a stray pen mark
        x0, y0 = rng.uniform(10, 74, 2)
        d = ImageDraw.Draw(big)
        d.line([(x0, y0), (x0 + rng.uniform(-12, 12), y0 + rng.uniform(-12, 12))], fill=255, width=4)
    big = big.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
    # crop to the ink, fit into 20x20 and centre in 28x28 (what MNIST does)
    bbox = big.getbbox() or (0, 0, 84, 84)
    crop = big.crop(bbox)
    k = 20 / max(crop.size)
    crop = crop.resize((max(1, int(crop.width * k)), max(1, int(crop.height * k))), Image.LANCZOS)
    out = Image.new("L", (28, 28), 0)
    out.paste(crop, ((28 - crop.width) // 2 + int(rng.integers(-1, 2)),
                     (28 - crop.height) // 2 + int(rng.integers(-1, 2))))
    return out


def make_mnist_sample(root, n_train=3000, n_valid=500, seed=2):
    """mnist_sample/{train,valid}/{3,7}/*.png (the 3s-versus-7s dataset of Chapter 4)."""
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("valid", n_valid)):
        for digit in (3, 7):
            out = root / split / str(digit)
            out.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                digit_image(digit, rng).save(out / f"{i}.png")
    with open(root / "labels.csv", "w") as f:
        f.write("name,label\n")
        for digit in (3, 7):
            for i in range(n_train):
                f.write(f"train/{digit}/{i}.png,{digit}\n")


def make_mnist(root, n_train=6000, n_test=1000, seed=3):
    """mnist_png/{training,testing}/<digit>/*.png (all ten digits, Chapter 13)."""
    rng = np.random.default_rng(seed)
    for split, n in (("training", n_train), ("testing", n_test)):
        for digit in range(10):
            out = root / split / str(digit)
            out.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                digit_image(digit, rng).save(out / f"{i}.png")


# --------------------------------------------------------------------------------------------------
# Object icons for PASCAL (multi-label) and Imagenette
# --------------------------------------------------------------------------------------------------
def _icon(d, name, x, y, s, rng):
    """Draw a small icon of `name` with centre (x, y) and size s."""
    J = lambda c: jitter(rng, c, 20)  # noqa: E731
    if name in ("person",):
        ellipse(d, x, y - s * 0.6, s * 0.2, s * 0.2, J((230, 190, 160)))
        d.rectangle([x - s * 0.22, y - s * 0.4, x + s * 0.22, y + s * 0.25], fill=J((50, 90, 180)))
        d.line([(x - s * 0.1, y + s * 0.25), (x - s * 0.2, y + s * 0.8)], fill=(40, 40, 40), width=max(2, int(s * .1)))
        d.line([(x + s * 0.1, y + s * 0.25), (x + s * 0.2, y + s * 0.8)], fill=(40, 40, 40), width=max(2, int(s * .1)))
    elif name in ("car", "bus", "garbage_truck"):
        L = {"car": 0.9, "bus": 1.3, "garbage_truck": 1.15}[name]
        col = J({"car": (200, 40, 40), "bus": (240, 190, 30), "garbage_truck": (60, 140, 70)}[name])
        top = y - s * (0.2 if name == "car" else 0.55)
        d.rectangle([x - s * L, top, x + s * L, y + s * 0.3], fill=col)
        if name == "car":
            d.rectangle([x - s * 0.45, y - s * 0.5, x + s * 0.45, top], fill=col)
        if name == "bus":
            for k in np.linspace(-L + 0.25, L - 0.25, 5):
                d.rectangle([x + s * (k - 0.12), y - s * 0.45, x + s * (k + 0.12), y - s * 0.15], fill=(200, 230, 250))
        if name == "garbage_truck":
            d.rectangle([x + s * 0.55, y - s * 0.3, x + s * L, y + s * 0.3], fill=(230, 230, 230))
        for k in (-L * 0.6, L * 0.6):
            ellipse(d, x + s * k, y + s * 0.35, s * 0.22, s * 0.22, (20, 20, 20))
    elif name in ("bicycle", "motorbike"):
        for k in (-0.5, 0.5):
            ellipse(d, x + s * k, y + s * 0.3, s * 0.32, s * 0.32, None, outline=(20, 20, 20), width=max(2, int(s * .08)))
        d.line([(x - s * 0.5, y + s * 0.3), (x, y - s * 0.1), (x + s * 0.5, y + s * 0.3)],
               fill=J((30, 30, 200)) if name == "bicycle" else J((220, 120, 0)), width=max(2, int(s * .1)))
        if name == "motorbike":
            d.rectangle([x - s * 0.25, y - s * 0.2, x + s * 0.25, y + s * 0.15], fill=J((220, 120, 0)))
    elif name in ("cat", "dog", "English_springer"):
        img = d._image
        spec = PET_BREEDS["Russian_Blue" if name == "cat" else "beagle"] if name != "English_springer" else \
            ("white", "patches", "brown", "floppy", 0.6, True)
        draw_animal(img, rng, *spec, x, y - s * 0.2, s * 0.45)
    elif name in ("bird",):
        ellipse(d, x, y, s * 0.45, s * 0.3, J((50, 120, 200)))
        ellipse(d, x + s * 0.4, y - s * 0.2, s * 0.18, s * 0.18, J((50, 120, 200)))
        poly(d, [(x + s * 0.55, y - s * 0.25), (x + s * 0.8, y - s * 0.18), (x + s * 0.55, y - s * 0.1)], (240, 180, 0))
        poly(d, [(x - s * 0.1, y - s * 0.1), (x - s * 0.5, y - s * 0.6), (x + s * 0.2, y - s * 0.15)], J((30, 80, 160)))
    elif name in ("boat",):
        poly(d, [(x - s * 0.8, y), (x + s * 0.8, y), (x + s * 0.55, y + s * 0.35), (x - s * 0.55, y + s * 0.35)], J((150, 80, 40)))
        poly(d, [(x, y - s * 0.9), (x, y - s * 0.05), (x + s * 0.6, y - s * 0.05)], J((245, 245, 245)))
    elif name in ("bottle",):
        d.rectangle([x - s * 0.2, y - s * 0.3, x + s * 0.2, y + s * 0.7], fill=J((40, 150, 70)))
        d.rectangle([x - s * 0.08, y - s * 0.7, x + s * 0.08, y - s * 0.3], fill=J((40, 150, 70)))
    elif name in ("chair",):
        c = J((140, 90, 40))
        d.rectangle([x - s * 0.4, y, x + s * 0.4, y + s * 0.1], fill=c)
        d.rectangle([x + s * 0.3, y - s * 0.7, x + s * 0.4, y + s * 0.7], fill=c)
        d.rectangle([x - s * 0.4, y, x - s * 0.3, y + s * 0.7], fill=c)
    elif name in ("cow", "sheep", "horse"):
        col = J({"cow": (240, 240, 240), "sheep": (230, 230, 215), "horse": (130, 80, 40)}[name])
        ellipse(d, x, y, s * 0.6, s * 0.32, col)
        ellipse(d, x + s * 0.65, y - s * 0.3, s * 0.2, s * 0.17, col if name != "sheep" else (40, 40, 40))
        for k in (-0.4, -0.15, 0.2, 0.45):
            d.line([(x + s * k, y + s * 0.2), (x + s * k, y + s * 0.7)], fill=darker(col, 0.5) if name != "sheep" else (40, 40, 40), width=max(2, int(s * .09)))
        if name == "cow":
            ellipse(d, x - s * 0.2, y - s * 0.05, s * 0.18, s * 0.12, (20, 20, 20))
        if name == "horse":
            d.line([(x - s * 0.6, y - s * 0.05), (x - s * 0.85, y + s * 0.35)], fill=(60, 30, 10), width=max(2, int(s * .1)))
    elif name in ("diningtable",):
        d.rectangle([x - s * 0.8, y - s * 0.1, x + s * 0.8, y + s * 0.05], fill=J((120, 70, 30)))
        for k in (-0.7, 0.65):
            d.rectangle([x + s * k, y, x + s * (k + 0.05), y + s * 0.6], fill=J((120, 70, 30)))
        ellipse(d, x - s * 0.3, y - s * 0.18, s * 0.15, s * 0.07, (250, 250, 250))
    elif name in ("pottedplant",):
        poly(d, [(x - s * 0.3, y + s * 0.2), (x + s * 0.3, y + s * 0.2), (x + s * 0.2, y + s * 0.7), (x - s * 0.2, y + s * 0.7)], J((180, 90, 50)))
        for a in np.linspace(-1.2, 1.2, 5):
            ellipse(d, x + math.sin(a) * s * 0.35, y - s * 0.15 - math.cos(a) * s * 0.3, s * 0.15, s * 0.25, J((40, 150, 40)))
    elif name in ("sofa",):
        c = J((150, 50, 120))
        d.rectangle([x - s * 0.9, y - s * 0.4, x + s * 0.9, y + s * 0.4], fill=c)
        d.rectangle([x - s * 0.7, y - s * 0.05, x + s * 0.7, y + s * 0.2], fill=darker(c, 0.8))
    elif name in ("train",):
        for k in (-1, 0, 1):
            d.rectangle([x + s * (k * 0.62 - 0.3), y - s * 0.4, x + s * (k * 0.62 + 0.3), y + s * 0.3], fill=J((80, 80, 160)))
        d.line([(x - s, y + s * 0.42), (x + s, y + s * 0.42)], fill=(60, 60, 60), width=max(2, int(s * .08)))
    elif name in ("tvmonitor",):
        d.rectangle([x - s * 0.6, y - s * 0.45, x + s * 0.6, y + s * 0.35], fill=(30, 30, 30))
        d.rectangle([x - s * 0.5, y - s * 0.37, x + s * 0.5, y + s * 0.27], fill=J((80, 160, 220)))
        d.rectangle([x - s * 0.1, y + s * 0.35, x + s * 0.1, y + s * 0.55], fill=(30, 30, 30))
    elif name in ("aeroplane",):
        d.rectangle([x - s * 0.9, y - s * 0.1, x + s * 0.9, y + s * 0.1], fill=J((210, 210, 220)))
        poly(d, [(x - s * 0.2, y), (x + s * 0.15, y), (x - s * 0.1, y - s * 0.7)], J((150, 150, 170)))
        poly(d, [(x - s * 0.2, y), (x + s * 0.15, y), (x - s * 0.1, y + s * 0.7)], J((150, 150, 170)))
        poly(d, [(x - s * 0.9, y), (x - s * 0.7, y), (x - s * 0.9, y - s * 0.4)], J((150, 150, 170)))
    # ---- Imagenette classes ----
    elif name == "tench":
        col = J((90, 120, 60))
        ellipse(d, x, y, s * 0.7, s * 0.3, col)
        poly(d, [(x - s * 0.6, y), (x - s * 1.0, y - s * 0.35), (x - s * 1.0, y + s * 0.35)], col)
        ellipse(d, x + s * 0.45, y - s * 0.07, s * 0.06, s * 0.06, (230, 150, 20))
        for k in np.linspace(-0.4, 0.3, 5):
            ellipse(d, x + s * k, y + s * 0.05, s * 0.06, s * 0.06, darker(col, 0.7))
    elif name == "cassette_player":
        d.rectangle([x - s * 0.8, y - s * 0.5, x + s * 0.8, y + s * 0.5], fill=J((60, 60, 70)))
        d.rectangle([x - s * 0.55, y - s * 0.3, x + s * 0.55, y + s * 0.15], fill=(200, 200, 190))
        for k in (-0.28, 0.28):
            ellipse(d, x + s * k, y - s * 0.07, s * 0.13, s * 0.13, (40, 40, 40))
        for k in np.linspace(-0.6, 0.6, 5):
            d.rectangle([x + s * (k - 0.08), y + s * 0.28, x + s * (k + 0.08), y + s * 0.42], fill=(180, 180, 180))
    elif name == "chain_saw":
        d.rectangle([x - s * 0.9, y - s * 0.4, x - s * 0.1, y + s * 0.3], fill=J((230, 110, 20)))
        d.rectangle([x - s * 0.1, y - s * 0.15, x + s * 1.0, y + s * 0.15], fill=(170, 170, 175))
        for k in np.linspace(-0.05, 0.95, 9):
            poly(d, [(x + s * k, y - s * 0.15), (x + s * (k + 0.06), y - s * 0.27), (x + s * (k + 0.12), y - s * 0.15)], (60, 60, 60))
    elif name == "church":
        c = J((200, 190, 170))
        d.rectangle([x - s * 0.7, y - s * 0.1, x + s * 0.4, y + s * 0.8], fill=c)
        d.rectangle([x + s * 0.4, y - s * 0.6, x + s * 0.8, y + s * 0.8], fill=c)
        poly(d, [(x + s * 0.35, y - s * 0.6), (x + s * 0.85, y - s * 0.6), (x + s * 0.6, y - s * 1.1)], J((120, 60, 50)))
        poly(d, [(x - s * 0.8, y - s * 0.1), (x + s * 0.5, y - s * 0.1), (x - s * 0.15, y - s * 0.5)], J((120, 60, 50)))
        d.line([(x + s * 0.6, y - s * 1.1), (x + s * 0.6, y - s * 1.35)], fill=(40, 40, 40), width=max(2, int(s * .06)))
        d.line([(x + s * 0.5, y - s * 1.25), (x + s * 0.7, y - s * 1.25)], fill=(40, 40, 40), width=max(2, int(s * .06)))
        d.rectangle([x - s * 0.25, y + s * 0.4, x - s * 0.05, y + s * 0.8], fill=(90, 60, 40))
    elif name == "French_horn":
        c = J((220, 170, 50))
        ellipse(d, x, y, s * 0.6, s * 0.6, None, outline=c, width=max(3, int(s * 0.15)))
        ellipse(d, x, y, s * 0.3, s * 0.3, None, outline=c, width=max(2, int(s * 0.1)))
        poly(d, [(x + s * 0.4, y + s * 0.2), (x + s * 1.0, y + s * 0.0), (x + s * 1.0, y + s * 0.8)], c)
    elif name == "gas_pump":
        c = J((200, 30, 30))
        d.rectangle([x - s * 0.35, y - s * 0.8, x + s * 0.35, y + s * 0.8], fill=c)
        d.rectangle([x - s * 0.25, y - s * 0.65, x + s * 0.25, y - s * 0.25], fill=(230, 240, 230))
        d.line([(x + s * 0.35, y - s * 0.1), (x + s * 0.7, y + s * 0.1), (x + s * 0.6, y + s * 0.6)], fill=(20, 20, 20), width=max(2, int(s * 0.08)))
    elif name == "golf_ball":
        ellipse(d, x, y + s * 0.6, s * 1.0, s * 0.25, J((60, 170, 60)))
        ellipse(d, x, y, s * 0.45, s * 0.45, (245, 245, 245))
        for _ in range(12):
            a, r = rng.uniform(0, 6.3), rng.uniform(0, 0.35)
            ellipse(d, x + math.cos(a) * r * s, y + math.sin(a) * r * s, s * 0.04, s * 0.04, (205, 205, 205))
    elif name == "parachute":
        c = J((240, 80, 160))
        d.pieslice([x - s * 0.9, y - s * 0.9, x + s * 0.9, y + s * 0.5], 180, 360, fill=c)
        for k in (-0.9, -0.3, 0.3, 0.9):
            d.line([(x + s * k, y - s * 0.2), (x, y + s * 0.7)], fill=(40, 40, 40), width=1)
        d.rectangle([x - s * 0.08, y + s * 0.65, x + s * 0.08, y + s * 0.9], fill=(40, 40, 40))
    else:
        raise ValueError(name)


PASCAL_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                  "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                  "train", "tvmonitor"]


def make_pascal(root, n_images=2500, seed=4):
    """pascal_2007/train/*.jpg + train.csv with space-separated labels and an is_valid column."""
    rng = np.random.default_rng(seed)
    (root / "train").mkdir(parents=True, exist_ok=True)
    common = ["person", "person", "person", "car", "dog", "cat", "chair", "bird"]   # imbalanced, like PASCAL
    rows = []
    for i in range(n_images):
        w, h = int(rng.integers(300, 500)), int(rng.integers(260, 400))
        img = background(rng, w, h)
        d = ImageDraw.Draw(img)
        n_obj = rng.choice([1, 1, 1, 2, 2, 3])
        labels = set()
        for _ in range(n_obj):
            name = (common[rng.integers(len(common))] if rng.random() < 0.4
                    else PASCAL_CLASSES[rng.integers(len(PASCAL_CLASSES))])
            s = min(w, h) * rng.uniform(0.12, 0.25)
            _icon(d, name, rng.uniform(s * 1.1, w - s * 1.1), rng.uniform(s * 1.1, h - s * 1.1), s, rng)
            labels.add(name)
        fname = f"{i:06d}.jpg"
        finish(img, rng).save(root / "train" / fname, quality=88)
        rows.append((fname, " ".join(sorted(labels)), rng.random() < 0.5))
    with open(root / "train.csv", "w") as f:
        f.write("fname,labels,is_valid\n")
        for fname, labels, valid in rows:
            f.write(f"{fname},{labels},{valid}\n")


IMAGENETTE = {"n01440764": "tench", "n02102040": "English_springer", "n02979186": "cassette_player",
              "n03000684": "chain_saw", "n03028079": "church", "n03394916": "French_horn",
              "n03417042": "garbage_truck", "n03425413": "gas_pump", "n03445777": "golf_ball",
              "n03888605": "parachute"}


def make_imagenette(root, n_train=260, n_valid=100, seed=5):
    """imagenette2-160/{train,val}/<wnid>/*.JPEG: 10 classes, shortest side 160 pixels."""
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("val", n_valid)):
        for wnid, name in IMAGENETTE.items():
            out = root / split / wnid
            out.mkdir(parents=True, exist_ok=True)
            others = [v for v in IMAGENETTE.values() if v != name]
            for i in range(n):
                w = int(rng.integers(160, 260))
                w, h = (w, 160) if rng.random() < 0.7 else (160, w)
                img = background(rng, w, h)
                d = ImageDraw.Draw(img)
                s = 160 * rng.uniform(0.14, 0.36)
                # like real photos, other things appear in the picture too (smaller than the main object)
                for _ in range(rng.choice([0, 1, 2, 2])):
                    s2 = s * rng.uniform(0.4, 0.8)
                    _icon(d, others[rng.integers(len(others))], rng.uniform(s2, w - s2), rng.uniform(s2, h - s2), s2, rng)
                _icon(d, name, rng.uniform(s, w - s), rng.uniform(s * 1.1, h - s * 0.9), s, rng)
                if rng.random() < 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img = img.rotate(rng.uniform(-20, 20), resample=Image.BILINEAR, fillcolor=jitter(rng, (128, 128, 128), 60))
                finish(img, rng, noise=14).save(out / f"{name}_{i}.JPEG", quality=85)


# --------------------------------------------------------------------------------------------------
# Head pose (regression) and street scenes (segmentation)
# --------------------------------------------------------------------------------------------------
_CAL = "517.679 0 320\n0 517.679 240.5\n0 0 1\n\n0 0 0 0\n\n1 0 0\n0 1 0\n0 0 1\n\n0 0 0\n"


def make_biwi(root, n_people=24, n_frames=120, seed=6):
    """biwi_head_pose/<person>/frame_<n>_rgb.jpg (480x640) + frame_<n>_pose.txt + rgb.cal.

    As in the real data, the pose file holds a rotation matrix and the 3D head centre in mm;
    projecting it with the camera calibration (rgb.cal) gives the head centre in pixels.
    """
    rng = np.random.default_rng(seed)
    for p in range(1, n_people + 1):
        person = root / f"{p:02d}"
        person.mkdir(parents=True, exist_ok=True)
        (person / "rgb.cal").write_text(_CAL)
        (root / f"{p:02d}.obj").write_text("# 3D head model (not used)\n")
        skin = jitter(rng, [(240, 200, 170), (200, 150, 110), (150, 100, 70), (100, 65, 45)][p % 4], 15)
        hair = jitter(rng, [(30, 25, 20), (120, 80, 40), (210, 180, 110), (90, 90, 90)][rng.integers(4)], 10)
        glasses = rng.random() < 0.35
        room = background(rng, 640, 480)
        for f in range(n_frames):
            img = room.copy()
            d = ImageDraw.Draw(img)
            z = rng.uniform(800, 1100)                          # distance from the camera, mm
            cx, cy = rng.uniform(170, 470), rng.uniform(150, 330)
            s = 517.679 * 95 / z                                # head radius in pixels
            yaw = rng.uniform(-0.5, 0.5)
            ellipse(d, cx, cy + s * 2.2, s * 1.6, s * 1.2, jitter(rng, (60, 80, 140)))       # shoulders
            ellipse(d, cx, cy - s * 0.25, s * 1.02, s * 1.05, hair)
            ellipse(d, cx + yaw * s * 0.3, cy + s * 0.05, s * 0.85, s * 1.0, skin)
            ex = cx + yaw * s * 0.6
            for side in (-1, 1):
                ellipse(d, ex + side * s * 0.35, cy - s * 0.1, s * 0.12, s * 0.08, (40, 30, 30))
                if glasses:
                    ellipse(d, ex + side * s * 0.35, cy - s * 0.1, s * 0.25, s * 0.2, None, outline=(20, 20, 20), width=3)
            poly(d, [(ex, cy - s * 0.05), (ex - s * 0.12, cy + s * 0.3), (ex + s * 0.12, cy + s * 0.3)], darker(skin, 0.8))
            d.line([(ex - s * 0.25, cy + s * 0.55), (ex + s * 0.25, cy + s * 0.55)], fill=(150, 60, 60), width=3)
            finish(img, rng, blur=False).save(person / f"frame_{f:05d}_rgb.jpg", quality=88)
            # head centre (cx, cy) in pixels -> 3D point in mm (inverse of get_ctr in the book)
            X, Y = (cx - 320) * z / 517.679, (cy - 240.5) * z / 517.679
            rot = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
            rows = "\n".join(" ".join(f"{v:.6f}" for v in r) for r in rot)
            (person / f"frame_{f:05d}_pose.txt").write_text(f"{rows}\n\n{X:.4f} {Y:.4f} {z:.4f}\n")


CAMVID_CODES = ["Animal", "Archway", "Bicyclist", "Bridge", "Building", "Car", "CartLuggagePram", "Child",
                "Column_Pole", "Fence", "LaneMkgsDriv", "LaneMkgsNonDriv", "Misc_Text", "MotorcycleScooter",
                "OtherMoving", "ParkingBlock", "Pedestrian", "Road", "RoadShoulder", "Sidewalk", "SignSymbol",
                "Sky", "SUVPickupTruck", "TrafficCone", "TrafficLight", "Train", "Tree", "Truck_Bus", "Tunnel",
                "VegetationMisc", "Void", "Wall"]


def make_camvid(root, n_images=200, h=96, w=128, seed=7):
    """camvid_tiny/images/*.png, labels/*_P.png (one class code per pixel) and codes.txt."""
    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "codes.txt").write_text("\n".join(CAMVID_CODES) + "\n")
    code = {c: i for i, c in enumerate(CAMVID_CODES)}
    for i in range(n_images):
        img = Image.new("RGB", (w, h))
        mask = Image.new("L", (w, h))
        di, dm = ImageDraw.Draw(img), ImageDraw.Draw(mask)

        def shape(kind, cls, box, color):
            getattr(di, kind)(box, fill=jitter(rng, color, 15))
            getattr(dm, kind)(box, fill=code[cls])

        horizon = int(h * rng.uniform(0.4, 0.55))
        shape("rectangle", "Sky", [0, 0, w, horizon], (130, 170, 220))
        x = -5
        while x < w:                                                    # buildings / trees on the skyline
            bw = rng.integers(15, 40)
            if rng.random() < 0.3:
                shape("ellipse", "Tree", [x, horizon - rng.integers(15, 35), x + bw, horizon + 8], (60, 120, 50))
            else:
                shape("rectangle", "Building", [x, horizon - rng.integers(10, 45), x + bw, horizon + 5],
                      [(150, 120, 100), (180, 170, 150), (120, 110, 120)][rng.integers(3)])
            x += bw
        shape("rectangle", "Sidewalk", [0, horizon, w, h], (170, 160, 150))
        shape("polygon", "Road", [(w * 0.42, horizon), (w * 0.58, horizon), (w * 1.1, h), (-w * 0.1, h)], (80, 80, 85))
        for t in np.linspace(0.15, 0.95, 4):                            # dashed lane markings
            y0 = horizon + (h - horizon) * t
            shape("rectangle", "LaneMkgsDriv", [w / 2 - 1 - 3 * t, y0, w / 2 + 1 + 3 * t, y0 + 4 * t + 1], (240, 240, 230))
        for _ in range(rng.integers(0, 4)):                             # cars
            cy = horizon + (h - horizon) * rng.uniform(0.2, 0.8)
            cs = 6 + 25 * (cy - horizon) / (h - horizon)
            cx = rng.uniform(cs, w - cs)
            shape("rectangle", "Car", [cx - cs, cy - cs * 0.6, cx + cs, cy + cs * 0.1],
                  [(200, 40, 40), (40, 60, 170), (220, 220, 220), (30, 30, 30)][rng.integers(4)])
        for _ in range(rng.integers(0, 3)):                             # pedestrians on the sidewalk
            px = rng.choice([rng.uniform(2, w * 0.25), rng.uniform(w * 0.75, w - 2)])
            py = horizon + (h - horizon) * rng.uniform(0.3, 0.9)
            ps = 4 + 10 * (py - horizon) / (h - horizon)
            shape("rectangle", "Pedestrian", [px - ps * 0.3, py - ps * 2, px + ps * 0.3, py], (90, 40, 120))
        if rng.random() < 0.7:                                          # a pole with a sign
            px = rng.choice([rng.uniform(5, 25), rng.uniform(w - 25, w - 5)])
            shape("rectangle", "Column_Pole", [px - 1, horizon - 30, px + 1, h - 5], (110, 110, 110))
            shape("rectangle", "SignSymbol", [px - 5, horizon - 32, px + 5, horizon - 24], (230, 200, 30))
        img = finish(img, rng, noise=5, blur=False)
        img.save(root / "images" / f"scene_{i:04d}.png")
        mask.save(root / "labels" / f"scene_{i:04d}_P.png")


def reset(root):
    if root.exists():
        shutil.rmtree(root)
