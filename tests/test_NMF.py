"""
Module for testing the NMF algorithm.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import pytest

from surprise.prediction_algorithms import NMF
from surprise.dataset import Dataset
from surprise.dataset import Reader
from surprise.evaluate import evaluate

# the test and train files are from the ml-100k dataset (10% of u1.base and
# 10 % of u1.test)
train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
data = Dataset.load_from_folds([(train_file, test_file)], Reader('ml-100k'))


def test_NMF_parameters():
    """Ensure that all parameters are taken into account."""

    # The baseline against which to compare.
    algo = NMF(n_factors=1, n_epochs=1)
    rmse_default = evaluate(algo, data, measures=['rmse'])['rmse']

    # n_factors
    algo = NMF(n_factors=2, n_epochs=1)
    rmse_factors = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_factors

    # n_epochs
    algo = NMF(n_factors=1, n_epochs=2)
    rmse_n_epochs = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_n_epochs

    # reg_u
    algo = NMF(n_factors=1, n_epochs=1, reg_u=1)
    rmse_reg_u = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_reg_u

    # reg_i
    algo = NMF(n_factors=1, n_epochs=1, reg_i=1)
    rmse_reg_i = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_reg_i

    # init_low
    algo = NMF(n_factors=1, n_epochs=1, init_low=.5)
    rmse_init_low = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_init_low

    # init_low
    with pytest.raises(ValueError):
        algo = NMF(n_factors=1, n_epochs=1, init_low=-1)

    # init_high
    algo = NMF(n_factors=1, n_epochs=1, init_high=.5)
    rmse_init_high = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_init_high
