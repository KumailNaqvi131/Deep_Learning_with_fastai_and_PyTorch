"""Shared by the Chapter 9 scripts: the Blue Book for Bulldozers data, prepared as in the book.

    df = pd.read_csv(path/'TrainAndValid.csv', low_memory=False)
    df['ProductSize'] = ordered categories 'Large' > 'Large / Medium' > ... > 'Compact'
    df[dep_var] = np.log(df[dep_var])                # the competition's metric is RMSE of log(price)
    df = add_datepart(df, 'saledate')
    cond = (df.saleYear<2011) | (df.saleMonth<10)   # validate on the last months (the future!)
    to = TabularPandas(df, [Categorify, FillMissing], cat, cont, y_names=dep_var, splits=splits)

With --real, the real Kaggle files must be in data/real/bluebook-for-bulldozers/ (see README).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import URLs, untar_data  # noqa: E402
from common.tabular import Categorify, FillMissing, TabularPandas, add_datepart, cont_cat_split  # noqa: E402

path = untar_data(URLs.BULLDOZERS)
dep_var = "SalePrice"
sizes = ["Large", "Large / Medium", "Medium", "Small", "Mini", "Compact"]


def load_df():
    df = pd.read_csv(path / "TrainAndValid.csv", low_memory=False)
    df["ProductSize"] = pd.Categorical(df["ProductSize"], categories=sizes, ordered=True)
    df[dep_var] = np.log(df[dep_var])
    return add_datepart(df, "saledate")


def get_splits(df):
    cond = (df.saleYear < 2011) | (df.saleMonth < 10)
    train_idx = np.where(cond)[0]
    valid_idx = np.where(~cond)[0]
    return list(train_idx), list(valid_idx)


def get_to(df=None):
    df = load_df() if df is None else df
    cont, cat = cont_cat_split(df, 1, dep_var=dep_var)
    return TabularPandas(df, [Categorify, FillMissing], cat, cont, y_names=dep_var, splits=get_splits(df))


def get_xs():
    """xs, y, valid_xs, valid_y with the YearMade=1000 placeholder replaced by 1950 (as in the book)."""
    to = get_to()
    xs, y = to.train.xs.copy(), to.train.y
    valid_xs, valid_y = to.valid.xs.copy(), to.valid.y
    xs.loc[xs["YearMade"] < 1900, "YearMade"] = 1950
    valid_xs.loc[valid_xs["YearMade"] < 1900, "YearMade"] = 1950
    return to, xs, y, valid_xs, valid_y


def r_mse(pred, y):
    return round(math.sqrt(((np.asarray(pred) - np.asarray(y)) ** 2).mean()), 6)


def m_rmse(m, xs, y):
    return r_mse(m.predict(xs), y)


def need_sklearn():
    try:
        import sklearn  # noqa: F401  (just checking that it is installed)
    except ImportError:
        print("This script needs scikit-learn:  pip install scikit-learn")
        raise SystemExit(0)


def rf(xs, y, n_estimators=40, max_samples=200_000, max_features=0.5, min_samples_leaf=5, **kwargs):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_jobs=-1, n_estimators=n_estimators, max_samples=min(max_samples, len(xs)),
                                 max_features=max_features, min_samples_leaf=min_samples_leaf,
                                 oob_score=True, random_state=42, **kwargs).fit(xs, y)


def rf_feat_importance(m, df):
    return pd.DataFrame({"cols": df.columns, "imp": m.feature_importances_}).sort_values("imp", ascending=False)


REDUNDANT = ["saleYear", "ProductGroupDesc", "fiBaseModel", "Grouser_Tracks"]   # found in 06_*.py


def get_final_xs():
    """The book's xs_final / valid_xs_final: only the important columns (importance > 0.005), minus
    the redundant ones. Cached in outputs/ because 06_feature_importance_and_redundancy.py computes it."""
    to, xs, y, valid_xs, valid_y = get_xs()
    cache = Path(__file__).parent / "outputs" / "final_columns.txt"
    if cache.exists():
        keep = cache.read_text().split()
    else:
        m = rf(xs, y)
        fi = rf_feat_importance(m, xs)
        keep = [c for c in fi[fi.imp > 0.005].cols if c not in REDUNDANT]
        cache.parent.mkdir(exist_ok=True)
        cache.write_text("\n".join(keep))
    return to, xs[keep], y, valid_xs[keep], valid_y
