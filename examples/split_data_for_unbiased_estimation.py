"""
This module descibes how to split a dataset into two parts A and B: A is for
tuning the algorithm parameters, and B is for having an unbiased estimation of
its performances. The tuning is done by Grid Search.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import random

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import GridSearch


# Load the full dataset.
data = Dataset.load_builtin('ml-100k')
raw_ratings = data.raw_ratings

# shuffle ratings if you want
random.shuffle(raw_ratings)

# A = 90% of the data, B = 10% of the data
threshold = int(.9 * len(raw_ratings))
A_raw_ratings = raw_ratings[:threshold]
B_raw_ratings = raw_ratings[threshold:]

data.raw_ratings = A_raw_ratings  # data is now the set A
data.split(n_folds=3)

# Select your best algo with grid search.
print('Grid Search...')
param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005]}
grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=0)
grid_search.evaluate(data)

algo = grid_search.best_estimator['RMSE']

# retrain on the whole set A
trainset = data.build_full_trainset()
algo.train(trainset)

# Compute biased accuracy on A
predictions = algo.test(trainset.build_testset())
print('Biased accuracy on A,', end='   ')
accuracy.rmse(predictions)

# Compute unbiased accuracy on B
testset = data.construct_testset(B_raw_ratings)  # testset is now the set B
predictions = algo.test(testset)
print('Unbiased accuracy on B,', end=' ')
accuracy.rmse(predictions)
