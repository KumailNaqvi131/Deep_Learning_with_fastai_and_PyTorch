"""
Chapter 1 — Your Deep Learning Journey
Topic 7: Beyond images: tabular data (book section "Deep Learning Is Not Just for Image
Classification")

Predict whether a person earns more than $50k a year from census data (age, education,
occupation, ...). The book's code:

    path = untar_data(URLs.ADULT_SAMPLE)
    dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
        cat_names = ['workclass', 'education', 'marital-status', 'occupation',
                     'relationship', 'race'],
        cont_names = ['age', 'fnlwgt', 'education-num'],
        procs = [Categorify, FillMissing, Normalize])
    learn = tabular_learner(dls, metrics=accuracy)
    learn.fit_one_cycle(3)

Categorical columns get embeddings, continuous ones are normalised (Chapter 9). There is no
pretrained model for tables, so we use fit_one_cycle instead of fine_tune.

Run:
    python 07_tabular_adult_income.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from common import *  # noqa: F401,F403,E402
from common.tabular import (Categorify, FillMissing, Normalize, TabularDataLoaders,  # noqa: E402
                            tabular_learner)

set_seed(42)
path = untar_data(URLs.ADULT_SAMPLE)
print(pd.read_csv(path / "adult.csv").head(), "\n")

dls = TabularDataLoaders.from_csv(path / "adult.csv", path=path, y_names="salary",
                                  cat_names=["workclass", "education", "marital-status", "occupation",
                                             "relationship", "race"],
                                  cont_names=["age", "fnlwgt", "education-num"],
                                  procs=[Categorify, FillMissing, Normalize])
dls.to.show(5)          # the processed table (categories shown by name)
learn = tabular_learner(dls, metrics=accuracy)
print(learn.model)
learn.fit_one_cycle(3)
