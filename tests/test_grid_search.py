"""
Module for testing SearchGrid class.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import random

import pytest
import numpy as np

from surprise import SVD
from surprise import KNNBaseline
from surprise import evaluate
from surprise import GridSearch


random.seed(0)

# Note: don't really know why but n_jobs must be set to 1, else deprecation
# warnings from data.folds() aren't raised.


def test_grid_search_cv_results(small_ml):
    """Ensure that the number of parameter combinations is correct."""
    param_grid = {'n_epochs': [1, 2], 'lr_all': [0.002, 0.005],
                  'reg_all': [0.4, 0.6], 'n_factors': [1], 'init_std_dev': [0]}
    with pytest.warns(UserWarning):
        grid_search = GridSearch(SVD, param_grid, n_jobs=1)
    with pytest.warns(UserWarning):
        grid_search.evaluate(small_ml)
    assert len(grid_search.cv_results['params']) == 8


def test_measure_is_not_case_sensitive(small_ml):
    """Ensure that all best_* dictionaries are case insensitive."""
    param_grid = {'n_epochs': [1], 'lr_all': [0.002, 0.005],
                  'reg_all': [0.4, 0.6], 'n_factors': [1], 'init_std_dev': [0]}
    with pytest.warns(UserWarning):
        grid_search = GridSearch(SVD, param_grid, measures=['FCP', 'mae',
                                                            'rMSE'], n_jobs=1)
    with pytest.warns(UserWarning):
        grid_search.evaluate(small_ml)
    assert grid_search.best_index['fcp'] == grid_search.best_index['FCP']
    assert grid_search.best_params['mAe'] == grid_search.best_params['MaE']
    assert grid_search.best_score['RmSE'] == grid_search.best_score['RMSE']


def test_best_estimator(small_ml):
    """Ensure that the best estimator is the one giving the best score (by
    re-running it)"""
    param_grid = {'n_epochs': [5], 'lr_all': [0.002, 0.005],
                  'reg_all': [0.4, 0.6], 'n_factors': [1], 'init_std_dev': [0]}
    with pytest.warns(UserWarning):
        grid_search = GridSearch(SVD, param_grid, measures=['FCP', 'mae',
                                                            'rMSE'], n_jobs=1)
    with pytest.warns(UserWarning):
        grid_search.evaluate(small_ml)
    best_estimator = grid_search.best_estimator['MAE']
    with pytest.warns(UserWarning):
        assert np.mean(evaluate(
            best_estimator, small_ml)['mae']) == grid_search.best_score['MAE']


def test_dict_parameters(small_ml):
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

    small_ml.split(2)

    with pytest.warns(UserWarning):
        grid_search = GridSearch(KNNBaseline, param_grid,
                                 measures=['FCP', 'mae', 'rMSE'], n_jobs=1)
    with pytest.warns(UserWarning):
        grid_search.evaluate(small_ml)
    assert len(grid_search.cv_results['params']) == 32


def test_same_splits(small_ml):
    """Ensure that all parameter combinations are tested on the same splits (we
    check that average RMSE scores are the same, which should be enough)."""

    small_ml.split(3)

    # all RMSE should be the same (as param combinations are the same)
    param_grid = {'n_epochs': [1, 1], 'lr_all': [.5, .5]}
    with pytest.warns(UserWarning):
        grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], n_jobs=-1)
    grid_search.evaluate(small_ml)

    rmse_scores = [s['RMSE'] for s in grid_search.cv_results['scores']]
    assert len(set(rmse_scores)) == 1  # assert rmse_scores are all equal

    # evaluate grid search again, to make sure that splits are still the same.
    grid_search.evaluate(small_ml)
    rmse_scores += [s['RMSE'] for s in grid_search.cv_results['scores']]
    assert len(set(rmse_scores)) == 1
