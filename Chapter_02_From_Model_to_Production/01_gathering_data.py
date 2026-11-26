"""
Chapter 2 — From Model to Production
Topic 1: Gathering data and removing broken files (book section "Gathering Data")

The book builds a bear classifier (grizzly / black / teddy) from a Bing image search:

    results = search_images_bing(key, 'grizzly bear')
    download_images(dest, urls=results.attrgot('contentUrl'))
    fns = get_image_files(path)
    failed = verify_images(fns)
    failed.map(Path.unlink)

Image search needs an API key, so by default this repo generates a synthetic bears dataset with
the same folder structure: bears/<grizzly|black|teddy>/*.jpg (150 per class). Like a real search
result it contains a few broken downloads, a few mislabelled pictures and two with no bear at all.

To use real images, put a text file with one image URL per line for each class, e.g.
grizzly.txt, black.txt, teddy.txt, in a folder and run:
    python 01_gathering_data.py --urls my_url_folder
The images are downloaded into data/real/bears/<class>/ (and later scripts use them with --real).

Run:
    python 01_gathering_data.py
"""

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import *  # noqa: F401,F403,E402

parser = argparse.ArgumentParser()
parser.add_argument("--urls", help="folder of <class>.txt files with one image URL per line")
parser.add_argument("--real", action="store_true", help="use data/real/bears instead of the synthetic data")
args = parser.parse_args()


def download_images(dest, urls, timeout=10):
    """Download every URL into `dest` as 00000000.jpg, 00000001.jpg, ... (skipping failures)."""
    dest.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(urls):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                (dest / f"{i:08d}.jpg").write_bytes(r.read())
        except Exception as e:
            print(f"  failed: {url} ({type(e).__name__})")


if args.urls:
    path = DATA_ROOT / "real" / "bears"
    for f in sorted(Path(args.urls).glob("*.txt")):
        urls = [u.strip() for u in f.read_text().splitlines() if u.strip()]
        print(f"downloading {len(urls)} images of '{f.stem}' ...")
        download_images(path / f.stem, urls)
elif args.real:
    path = DATA_ROOT / "real" / "bears"
else:
    path = untar_data(URLs.BEARS)

fns = get_image_files(path)
print(f"{len(fns)} image files in {path}")
for kind in sorted(p.name for p in path.iterdir() if p.is_dir()):
    print(f"  {kind:8s}: {len(list((path / kind).glob('*')))} files")

# Some downloads are not images (an HTML error page, a truncated file). Find them and delete them.
failed = verify_images(fns)
print(f"\n{len(failed)} files could not be opened as images:", [f.name for f in failed])
for f in failed:
    f.unlink()
print(f"Deleted them. {len(get_image_files(path))} images left.")

show_images([PILImage.create(f) for f in get_image_files(path)[::60][:8]],
            [f.parent.name for f in get_image_files(path)[::60][:8]], outputs_dir(__file__) / "some_bears.png")
