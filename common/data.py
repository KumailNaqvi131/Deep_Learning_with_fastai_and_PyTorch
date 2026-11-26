"""`untar_data(URLs.X)`: the book's one-line way to get a dataset, working offline.

    path = untar_data(URLs.PETS)          # -> data/oxford-iiit-pet  (synthetic, generated once)
    python some_script.py --real          # -> data/real/oxford-iiit-pet (downloaded from fast.ai)

By default every dataset is a synthetic look-alike generated on first use (a few seconds,
no internet needed). With `--real` (or the environment variable REAL_DATA=1) the real dataset
is downloaded from the same URL fastai uses and extracted under data/real/.
"""

import shutil
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

from . import synth_data, synth_vision
from .utils import DATA_ROOT, use_real_data


class URLs:
    """Dataset names (fastai's URLs.X constants point at download links; here they are just names)."""
    PETS = "oxford-iiit-pet"
    BEARS = "bears"
    MNIST_SAMPLE = "mnist_sample"
    MNIST = "mnist_png"
    PASCAL_2007 = "pascal_2007"
    BIWI_HEAD_POSE = "biwi_head_pose"
    IMAGENETTE = "imagenette2-160"           # the book uses full-size Imagenette; we use the 160px version
    IMAGENETTE_160 = "imagenette2-160"
    CAMVID_TINY = "camvid_tiny"
    ADULT_SAMPLE = "adult_sample"
    ML_SAMPLE = "movie_lens_sample"
    ML_100k = "ml-100k"
    IMDB = "imdb"
    WIKITEXT = "wikitext_synth"
    HUMAN_NUMBERS = "human_numbers"
    BULLDOZERS = "bluebook-for-bulldozers"


S3 = "https://s3.amazonaws.com/"
REAL_URLS = {
    URLs.PETS: S3 + "fast-ai-imageclas/oxford-iiit-pet.tgz",                 # 800 MB
    URLs.MNIST_SAMPLE: S3 + "fast-ai-sample/mnist_sample.tgz",               # 3 MB
    URLs.MNIST: S3 + "fast-ai-imageclas/mnist_png.tgz",                      # 15 MB
    URLs.PASCAL_2007: S3 + "fast-ai-imagelocal/pascal_2007.tgz",             # 1.6 GB
    URLs.BIWI_HEAD_POSE: S3 + "fast-ai-imagelocal/biwi_head_pose.tgz",       # 450 MB
    URLs.IMAGENETTE: S3 + "fast-ai-imageclas/imagenette2-160.tgz",           # 95 MB
    URLs.CAMVID_TINY: S3 + "fast-ai-sample/camvid_tiny.tgz",                 # 2 MB
    URLs.ADULT_SAMPLE: S3 + "fast-ai-sample/adult_sample.tgz",               # 0.5 MB
    URLs.ML_SAMPLE: S3 + "fast-ai-sample/movie_lens_sample.tgz",             # 0.05 MB
    URLs.ML_100k: "https://files.grouplens.org/datasets/movielens/ml-100k.zip",  # 5 MB
    URLs.IMDB: S3 + "fast-ai-nlp/imdb.tgz",                                  # 145 MB
    URLs.HUMAN_NUMBERS: S3 + "fast-ai-sample/human_numbers.tgz",             # 0.03 MB
}
NO_REAL_VERSION = {
    URLs.BEARS: "The book builds this dataset with a Bing image search. Put your own images in "
                "data/real/bears/<class>/ to use real photos.",
    URLs.BULLDOZERS: "Download it from Kaggle (kaggle competitions download -c bluebook-for-bulldozers) "
                     "and unzip it into data/real/bluebook-for-bulldozers/.",
    URLs.WIKITEXT: "There is no real version: this text only stands in for the WikiText-103 pre-training.",
}


def _generate(name, dest):
    sv, sd = synth_vision, synth_data
    gens = {
        URLs.PETS: lambda: sv.make_pets(dest),
        URLs.BEARS: lambda: sv.make_bears(dest),
        URLs.MNIST_SAMPLE: lambda: sv.make_mnist_sample(dest),
        URLs.MNIST: lambda: sv.make_mnist(dest),
        URLs.PASCAL_2007: lambda: sv.make_pascal(dest),
        URLs.BIWI_HEAD_POSE: lambda: sv.make_biwi(dest),
        URLs.IMAGENETTE: lambda: sv.make_imagenette(dest),
        URLs.CAMVID_TINY: lambda: sv.make_camvid(dest),
        URLs.ADULT_SAMPLE: lambda: sd.make_adult(dest),
        URLs.ML_100k: lambda: sd.make_movielens(dest),
        URLs.ML_SAMPLE: lambda: sd.make_ml_sample(dest, untar_data(URLs.ML_100k, real=False)),
        URLs.IMDB: lambda: sd.make_imdb(dest),
        URLs.WIKITEXT: lambda: sd.make_wikitext(dest),
        URLs.HUMAN_NUMBERS: lambda: sd.make_human_numbers(dest),
        URLs.BULLDOZERS: lambda: sd.make_bulldozers(dest),
    }
    print(f"[data] generating synthetic '{name}' in {dest} (first time only) ...", flush=True)
    t0 = time.time()
    dest.mkdir(parents=True, exist_ok=True)
    gens[name]()
    print(f"[data] done in {time.time() - t0:.1f}s", flush=True)


def _download(name, dest):
    url = REAL_URLS[name]
    archive = dest.parent / Path(url).name
    print(f"[data] downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, archive)
    print("[data] extracting ...", flush=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest.parent)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(dest.parent, filter="data")        # Python >= 3.11.4: safe extraction
            except TypeError:
                t.extractall(dest.parent)
    archive.unlink()


def untar_data(name: str, real: bool = None) -> Path:
    """Return the folder of dataset `name`, generating (default) or downloading (`--real`) it first."""
    real = use_real_data() if real is None else real
    if real:
        dest = DATA_ROOT / "real" / name
        if dest.exists():
            return dest
        if name in NO_REAL_VERSION:
            raise SystemExit(f"No automatic download for '{name}'. {NO_REAL_VERSION[name]}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        _download(name, dest)
        return dest
    dest = DATA_ROOT / name
    if not (dest / ".complete").exists():           # a marker, so an interrupted run starts over
        if dest.exists():
            shutil.rmtree(dest)
        _generate(name, dest)
        (dest / ".complete").write_text("ok\n")
    return dest
