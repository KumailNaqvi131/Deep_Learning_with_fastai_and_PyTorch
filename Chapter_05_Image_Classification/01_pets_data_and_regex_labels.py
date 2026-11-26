"""
Chapter 5 — Image Classification
Topic 1: From dogs and cats to pet breeds: getting labels with a regular expression (book
section "From Dogs and Cats to Pet Breeds")

Now we want the *breed* (37 classes), not just cat vs dog. The breed is in the file name:
great_pyrenees_173.jpg -> great_pyrenees. A regular expression (regex) extracts it:

    re.findall(r'(.+)_\\d+.jpg$', fname.name)
      (.+)     one or more characters, captured: the breed
      _\\d+     an underscore followed by digits
      .jpg$    the extension at the end of the name

Run:
    python 01_pets_data_and_regex_labels.py        # add --real for the real Oxford-IIIT Pet dataset
"""

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import RegexLabeller, URLs, get_image_files, untar_data, using_attr  # noqa: E402

path = untar_data(URLs.PETS)
print("path.ls():", sorted(p.name for p in path.iterdir()))
files = get_image_files(path / "images")
print(f"{len(files)} images, for example: {[f.name for f in files[:3]]}")

fname = files[0]
print(f"\nre.findall(r'(.+)_\\d+.jpg$', '{fname.name}') ->", re.findall(r"(.+)_\d+.jpg$", fname.name))

labeller = using_attr(RegexLabeller(r"(.+)_\d+.jpg$"), "name")   # what the DataBlock uses as get_y
labels = [labeller(f) for f in files]
counts = Counter(labels)
print(f"\n{len(counts)} breeds; images per breed: min {min(counts.values())}, max {max(counts.values())}")
cats = sorted(b for b in counts if b[0].isupper())
dogs = sorted(b for b in counts if b[0].islower())
print(f"cat breeds ({len(cats)}):", ", ".join(cats))
print(f"dog breeds ({len(dogs)}):", ", ".join(dogs))
