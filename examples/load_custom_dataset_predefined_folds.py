"""
This module descibes how to load a custom dataset when folds (for
cross-validation) are predefined by train and test files.

As a custom dataset we will actually use the movielens-100k dataset, but act as
if it were not built-in.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import PredefinedKFold

# path to dataset folder
files_dir = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/')

# This time, we'll use the built-in reader.
reader = Reader('ml-100k')

# folds_files is a list of tuples containing file paths:
# [(u1.base, u1.test), (u2.base, u2.test), ... (u5.base, u5.test)]
train_file = files_dir + 'u%d.base'
test_file = files_dir + 'u%d.test'
folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

data = Dataset.load_from_folds(folds_files, reader=reader)
pkf = PredefinedKFold()

algo = SVD()

for trainset, testset in pkf.split(data):

    # train and test algorithm.
    algo.fit(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    accuracy.rmse(predictions, verbose=True)
