"""
Module for testing SearchGrid class.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import random

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import KNNBaseline
from surprise import evaluate
from surprise import GridSearch

# the test and train files are from the ml-100k dataset (10% of u1.base and
# 10 % of u1.test)
train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
data = Dataset.load_from_folds([(train_file, test_file)], Reader('ml-100k'))

random.seed(0)


def test_grid_search_cv_results():
    """Ensure that the number of parameter combinations is correct."""
    param_grid = {'n_epochs': [1, 2], 'lr_all': [0.002, 0.005],
                  'reg_all': [0.4, 0.6], 'n_factors': [1], 'init_std_dev': [0]}
    grid_search = GridSearch(SVD, param_grid)
    grid_search.evaluate(data)
    assert len(grid_search.cv_results['params']) == 8


def test_measure_is_not_case_sensitive():
    """Ensure that all best_* dictionaries are case insensitive."""
    param_grid = {'n_epochs': [1], 'lr_all': [0.002, 0.005],
                  'reg_all': [0.4, 0.6], 'n_factors': [1], 'init_std_dev': [0]}
    grid_search = GridSearch(SVD, param_grid, measures=['FCP', 'mae', 'rMSE'])
    grid_search.evaluate(data)
    assert grid_search.best_index['fcp'] == grid_search.best_index['FCP']
    assert grid_search.best_params['mAe'] == grid_search.best_params['MaE']
    assert grid_search.best_score['RmSE'] == grid_search.best_score['RMSE']


def test_best_estimator():
    """Ensure that the best estimator is the one giving the best score (by
    re-running it)"""
    param_grid = {'n_epochs': [5], 'lr_all': [0.002, 0.005],
                  'reg_all': [0.4, 0.6], 'n_factors': [1], 'init_std_dev': [0]}
    grid_search = GridSearch(SVD, param_grid, measures=['FCP', 'mae', 'rMSE'])
    grid_search.evaluate(data)
    best_estimator = grid_search.best_estimator['MAE']
    assert evaluate(
        best_estimator, data)['MAE'] == grid_search.best_score['MAE']


def test_dict_parameters():
    """Dict parameters like bsl_options and sim_options require special
    treatment in the param_grid argument. We here test both in one shot with
    KNNBaseline."""

    param_grid = {'bsl_options': {'method': ['als', 'sgd'],
                                  'reg': [1, 2]},
                  'k': [2, 3],
                  'sim_options': {'name': ['msd', 'cosine'],
                                  'min_support': [1, 5],
                                  'user_based': [False]}
                  }

    grid_search = GridSearch(KNNBaseline, param_grid,
                             measures=['FCP', 'mae', 'rMSE'])
    grid_search.evaluate(data)
    assert len(grid_search.cv_results['params']) == 32


def test_same_splits():
    """Ensure that all parameter combinations are tested on the same splits (we
    check that average RMSE scores are the same, which should be enough)."""

    data_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
    data = Dataset.load_from_file(data_file, reader=Reader('ml-100k'))
    data.split(3)

    # all RMSE should be the same (as param combinations are the same)
    param_grid = {'n_epochs': [1, 1], 'lr_all': [.5, .5]}
    grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], n_jobs=-1)
    grid_search.evaluate(data)

    rmse_scores = [s['RMSE'] for s in grid_search.cv_results['scores']]
    assert len(set(rmse_scores)) == 1  # assert rmse_scores are all equal

    # evaluate grid search again, to make sure that splits are still the same.
    grid_search.evaluate(data)
    rmse_scores += [s['RMSE'] for s in grid_search.cv_results['scores']]
    assert len(set(rmse_scores)) == 1
