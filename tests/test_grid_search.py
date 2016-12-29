"""
Module for testing SearchGrid class.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

from surprise.evaluate import GridSearch
from surprise.dataset import Dataset
from surprise.dataset import Reader
from surprise.prediction_algorithms import SVD

# the test and train files are from the ml-100k dataset (10% of u1.base and
# 10 % of u1.test)
train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
data = Dataset.load_from_folds([(train_file, test_file)], Reader('ml-100k'))

def test_grid_search():
    param_grid = {'n_epochs':[5,10], 'lr_all': [0.002, 0.005], 'reg_all': [0.4,0.6]}
    gridSearch = GridSearch(SVD,param_grid)
    gridSearch.evaluate(data)
    assert len(gridSearch.combination_score_list) == 8