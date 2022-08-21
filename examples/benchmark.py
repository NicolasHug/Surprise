"""This module runs a 5-Fold CV for all the algorithms (default parameters) on
the movielens datasets, and reports average RMSE, MAE, and total computation
time.  It is used for making tables in the README.md file"""

# flake8: noqa

import datetime
import random
import time

import numpy as np

from surprise import (
    BaselineOnly,
    CoClustering,
    Dataset,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    NMF,
    NormalPredictor,
    SlopeOne,
    SVD,
    SVDpp,
)
from surprise.model_selection import cross_validate, KFold
from tabulate import tabulate

# The algorithms to cross-validate
algos = (
    SVD(random_state=0),
    SVDpp(random_state=0, cache_ratings=False),
    SVDpp(random_state=0, cache_ratings=True),
    NMF(random_state=0),
    SlopeOne(),
    KNNBasic(),
    KNNWithMeans(),
    KNNBaseline(),
    CoClustering(random_state=0),
    BaselineOnly(),
    NormalPredictor(),
)

# ugly dict to map algo names and datasets to their markdown links in the table
stable = "https://surprise.readthedocs.io/en/stable/"
LINK = {
    "SVD": "[{}]({})".format(
        "SVD",
        stable
        + "matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD",
    ),
    "SVDpp": "[{}]({})".format(
        "SVD++",
        stable
        + "matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp",
    ),
    "NMF": "[{}]({})".format(
        "NMF",
        stable
        + "matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF",
    ),
    "SlopeOne": "[{}]({})".format(
        "Slope One",
        stable + "slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne",
    ),
    "KNNBasic": "[{}]({})".format(
        "k-NN",
        stable + "knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic",
    ),
    "KNNWithMeans": "[{}]({})".format(
        "Centered k-NN",
        stable + "knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans",
    ),
    "KNNBaseline": "[{}]({})".format(
        "k-NN Baseline",
        stable + "knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline",
    ),
    "CoClustering": "[{}]({})".format(
        "Co-Clustering",
        stable
        + "co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering",
    ),
    "BaselineOnly": "[{}]({})".format(
        "Baseline",
        stable
        + "basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly",
    ),
    "NormalPredictor": "[{}]({})".format(
        "Random",
        stable
        + "basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor",
    ),
    "ml-100k": "[{}]({})".format(
        "Movielens 100k", "https://grouplens.org/datasets/movielens/100k"
    ),
    "ml-1m": "[{}]({})".format(
        "Movielens 1M", "https://grouplens.org/datasets/movielens/1m"
    ),
}


# set RNG
np.random.seed(0)
random.seed(0)

dataset = "ml-100k"
data = Dataset.load_builtin(dataset)
kf = KFold(random_state=0)  # folds will be the same for all algorithms.

table = []
for algo in algos:
    start = time.time()
    out = cross_validate(algo, data, ["rmse", "mae"], kf)
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
    link = LINK[algo.__class__.__name__]
    mean_rmse = "{:.3f}".format(np.mean(out["test_rmse"]))
    mean_mae = "{:.3f}".format(np.mean(out["test_mae"]))

    new_line = [link, mean_rmse, mean_mae, cv_time]
    print(tabulate([new_line], tablefmt="pipe"))  # print current algo perf
    table.append(new_line)

header = [LINK[dataset], "RMSE", "MAE", "Time"]
print(tabulate(table, header, tablefmt="pipe"))
