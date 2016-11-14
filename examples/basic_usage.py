"""
This module descibes the most basic usage of RecSys: you define a prediction
algorithm, (down)load a dataset and evaluate the performances of the algorithm.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from recsys import SVD
from recsys import Dataset
from recsys import evaluate


# Load the movielens-100k dataset and split it into 3 folds for
# cross-validation.
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print(perf['RMSE'])
print(perf['MAE'])
