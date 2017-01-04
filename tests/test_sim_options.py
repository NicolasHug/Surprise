"""
Module for testing the sim_options parameter.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
from itertools import combinations

import pytest

from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader
from surprise import evaluate


# the test and train files are from the ml-100k dataset (10% of u1.base and
# 10 % of u1.test)
train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
data = Dataset.load_from_folds([(train_file, test_file)], Reader('ml-100k'))


def test_name_field():
    """Ensure the name field is taken into account."""

    sim_options = {'name': 'cosine'}
    algo = KNNBasic(sim_options=sim_options)
    rmse_cosine = evaluate(algo, data, measures=['rmse'])['rmse']

    sim_options = {'name': 'msd'}
    algo = KNNBasic(sim_options=sim_options)
    rmse_msd = evaluate(algo, data, measures=['rmse'])['rmse']

    sim_options = {'name': 'pearson'}
    algo = KNNBasic(sim_options=sim_options)
    rmse_pearson = evaluate(algo, data, measures=['rmse'])['rmse']

    sim_options = {'name': 'pearson_baseline'}
    bsl_options = {'n_epochs': 1}
    algo = KNNBasic(sim_options=sim_options, bsl_options=bsl_options)
    rmse_pearson_bsl = evaluate(algo, data, measures=['rmse'])['rmse']

    for rmse_a, rmse_b in combinations((rmse_cosine, rmse_msd, rmse_pearson,
                                        rmse_pearson_bsl), 2):
        assert (rmse_a != rmse_b)

    with pytest.raises(NameError):
        sim_options = {'name': 'wrong_name'}
        algo = KNNBasic(sim_options=sim_options)
        evaluate(algo, data)


def test_user_based_field():
    """Ensure that the user_based field is taken into account (only) when
    needed."""

    algorithms = (KNNBasic, KNNWithMeans, KNNBaseline)
    for klass in algorithms:
        algo = klass(sim_options={'user_based': True})
        rmses_user_based = evaluate(algo, data, measures=['rmse'])['rmse']
        algo = klass(sim_options={'user_based': False})
        rmses_item_based = evaluate(algo, data, measures=['rmse'])['rmse']
        assert rmses_user_based != rmses_item_based


def test_shrinkage_field():
    """Ensure the shrinkage field is taken into account."""

    sim_options = {'name': 'pearson_baseline',
                   'shrinkage': 0
                   }
    bsl_options = {'n_epochs': 1}
    algo = KNNBasic(sim_options=sim_options)
    rmse_shrinkage_0 = evaluate(algo, data, measures=['rmse'])['rmse']

    sim_options = {'name': 'pearson_baseline',
                   'shrinkage': 100
                   }
    bsl_options = {'n_epochs': 1}
    algo = KNNBasic(sim_options=sim_options, bsl_options=bsl_options)
    rmse_shrinkage_100 = evaluate(algo, data, measures=['rmse'])['rmse']

    assert rmse_shrinkage_0 != rmse_shrinkage_100
