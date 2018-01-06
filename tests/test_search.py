"""
Module for testing the model_selection.search module.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

import numpy as np

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import KNNBasic
from surprise.model_selection import KFold
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate


def test_parameter_combinations():
    """Make sure that parameter_combinations attribute is correct (has correct
    size).  Dict parameters like bsl_options and sim_options require special
    treatment in the param_grid argument. We here test both in one shot with
    KNNBaseline."""

    param_grid = {'bsl_options': {'method': ['als', 'sgd'],
                                  'reg': [1, 2]},
                  'k': [2, 3],
                  'sim_options': {'name': ['msd', 'cosine'],
                                  'min_support': [1, 5],
                                  'user_based': [False]}
                  }

    gs = GridSearchCV(SVD, param_grid)
    assert len(gs.param_combinations) == 32


def test_best_estimator():
    """Ensure that the best estimator is the one giving the best score (by
    re-running it)"""

    train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
    test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
    data = Dataset.load_from_folds([(train_file, test_file)],
                                   Reader('ml-100k'))

    param_grid = {'n_epochs': [5], 'lr_all': [0.002, 0.005],
                  'reg_all': [0.4, 0.6], 'n_factors': [1], 'init_std_dev': [0]}
    gs = GridSearchCV(SVD, param_grid, measures=['mae'],
                      cv=PredefinedKFold(), joblib_verbose=100)
    gs.fit(data)
    best_estimator = gs.best_estimator['mae']

    # recompute MAE of best_estimator
    mae = cross_validate(best_estimator, data, measures=['MAE'],
                         cv=PredefinedKFold())['test_mae']

    assert mae == gs.best_score['mae']


def test_same_splits():
    """Ensure that all parameter combinations are tested on the same splits (we
    check their RMSE scores are the same once averaged over the splits, which
    should be enough). We use as much parallelism as possible."""

    data_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
    data = Dataset.load_from_file(data_file, reader=Reader('ml-100k'))
    kf = KFold(3, shuffle=True, random_state=4)

    # all RMSE should be the same (as param combinations are the same)
    param_grid = {'k': [1, 1], 'min_k': [3, 3]}
    gs = GridSearchCV(KNNBasic, param_grid, measures=['RMSE'], cv=kf,
                      n_jobs=-1)
    gs.fit(data)

    rmse_scores = [m for m in gs.cv_results['mean_test_rmse']]
    assert len(set(rmse_scores)) == 1  # assert rmse_scores are all equal

    # Note: actually, even when setting random_state=None in kf, the same folds
    # are used because we use product(param_comb, kf.split(...)). However, it's
    # needed to have the same folds when calling fit again:
    gs.fit(data)
    rmse_scores += [m for m in gs.cv_results['mean_test_rmse']]
    assert len(set(rmse_scores)) == 1  # assert rmse_scores are all equal


def test_cv_results():
    '''Test the cv_results attribute'''

    f = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
    data = Dataset.load_from_file(f, Reader('ml-100k'))
    kf = KFold(3, shuffle=True, random_state=4)
    param_grid = {'k': [1, 10], 'sim_options': {'name': ['msd', 'cosine']}}
    gs = GridSearchCV(KNNBasic, param_grid, measures=['RMSE', 'mae'], cv=kf)
    gs.fit(data)

    # test keys split*_test_rmse, mean and std dev.
    assert gs.cv_results['split0_test_rmse'].shape == (4,)  # 4 param comb.
    assert gs.cv_results['split1_test_rmse'].shape == (4,)  # 4 param comb.
    assert gs.cv_results['split2_test_rmse'].shape == (4,)  # 4 param comb.
    assert gs.cv_results['mean_test_rmse'].shape == (4,)  # 4 param comb.
    assert np.allclose(gs.cv_results['mean_test_rmse'],
                       np.mean([gs.cv_results['split0_test_rmse'],
                                gs.cv_results['split1_test_rmse'],
                                gs.cv_results['split2_test_rmse']], axis=0))
    assert np.allclose(gs.cv_results['std_test_rmse'],
                       np.std([gs.cv_results['split0_test_rmse'],
                               gs.cv_results['split1_test_rmse'],
                               gs.cv_results['split2_test_rmse']], axis=0))

    # test fit and train times dimensions.
    assert gs.cv_results['mean_fit_time'].shape == (4,)  # 4 param comb.
    assert gs.cv_results['std_fit_time'].shape == (4,)  # 4 param comb.
    assert gs.cv_results['mean_test_time'].shape == (4,)  # 4 param comb.
    assert gs.cv_results['std_test_time'].shape == (4,)  # 4 param comb.

    assert gs.cv_results['params'] is gs.param_combinations

    # assert that best parameter in gs.cv_results['rank_test_measure'] is
    # indeed the best_param attribute.
    best_index = np.argmin(gs.cv_results['rank_test_rmse'])
    assert gs.cv_results['params'][best_index] == gs.best_params['rmse']
    best_index = np.argmin(gs.cv_results['rank_test_mae'])
    assert gs.cv_results['params'][best_index] == gs.best_params['mae']
