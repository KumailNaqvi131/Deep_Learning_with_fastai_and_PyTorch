"""Shared by the Chapter 8 scripts: MovieLens 100k ratings with movie titles, and its DataLoaders.

    ratings = pd.read_csv(path/'u.data', delimiter='\\t', header=None,
                          names=['user','movie','rating','timestamp'])
    movies = pd.read_csv(path/'u.item', delimiter='|', encoding='latin-1',
                         usecols=(0,1), names=('movie','title'), header=None)
    ratings = ratings.merge(movies)
    dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common import URLs, untar_data  # noqa: E402
from common.collab import CollabDataLoaders  # noqa: E402

path = untar_data(URLs.ML_100k)
ratings = pd.read_csv(path / "u.data", delimiter="\t", header=None, names=["user", "movie", "rating", "timestamp"])
movies = pd.read_csv(path / "u.item", delimiter="|", encoding="latin-1", usecols=(0, 1), names=("movie", "title"), header=None)
ratings = ratings.merge(movies)


def get_dls(bs=64, seed=42):
    return CollabDataLoaders.from_df(ratings, item_name="title", bs=bs, seed=seed)
