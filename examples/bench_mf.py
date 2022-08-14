"""This module runs a 5-Fold CV for all the algorithms (default parameters) on
the movielens datasets, and reports average RMSE, MAE, and total computation
time.  It is used for making tables in the README.md file"""

from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import random
import time

import numpy as np

from surprise import Dataset, NMF, SVD, SVDpp
from surprise.model_selection import cross_validate, KFold
from tabulate import tabulate

# The algorithms to cross-validate
classes = (SVD, SVDpp, NMF)

# ugly dict to map algo names and datasets to their markdown links in the table
stable = "http://surprise.readthedocs.io/en/stable/"
mf_docs = (
    "matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization."
)

LINK = {
    "SVD": "[{}]({})".format("SVD", stable + mf_docs + "SVD"),
    "SVDpp": "[{}]({})".format("SVD++", stable + mf_docs + "SVDpp"),
    "NMF": "[{}]({})".format("NMF", stable + mf_docs + "NMF"),
    "ml-100k": "[{}]({})".format(
        "Movielens 100k",
        "http://grouplens.org/datasets/movielens/100k",
    ),
    "ml-1m": "[{}]({})".format(
        "Movielens 1M",
        "http://grouplens.org/datasets/movielens/1m",
    ),
}


# set RNG
np.random.seed(0)
random.seed(0)

dataset = "ml-100k"
data = Dataset.load_builtin(dataset)
kf = KFold(random_state=0)  # folds will be the same for all algorithms.

table = []
for klass in classes:
    start = time.time()
    out = cross_validate(klass(), data, ["rmse", "mae"], kf)
    cv_time = str(datetime.timedelta(seconds=int(time.time() - start)))
    link = LINK[klass.__name__]
    mean_rmse = "{:.3f}".format(np.mean(out["test_rmse"]))
    mean_mae = "{:.3f}".format(np.mean(out["test_mae"]))

    new_line = [link, mean_rmse, mean_mae, cv_time]
    print(tabulate([new_line], tablefmt="pipe"))  # print current algo perf
    table.append(new_line)

header = [LINK[dataset], "RMSE", "MAE", "Time"]
print(tabulate(table, header, tablefmt="pipe"))
