"""
Module for testing the SVD algorithm.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

from recsys.prediction_algorithms import SVD
from recsys.dataset import Dataset
from recsys.dataset import Reader
from recsys.evaluate import evaluate


# the test and train files are from the ml-100k dataset (10% of u1.base and
# 10 % of u1.test)
train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
data = Dataset.load_from_folds([(train_file, test_file)], Reader('ml-100k'))

def test_parameters():
    """Ensure that all parameters are taken into account."""

    # The baseline against which to compare.
    algo = SVD(n_factors=1, n_epochs=1)
    rmse_default = evaluate(algo, data, measures=['rmse'])['rmse']

    # n_factors
    algo = SVD(n_factors=2, n_epochs=1)
    rmse_factors = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_factors

    # n_epochs
    algo = SVD(n_factors=1, n_epochs=2)
    rmse_n_epochs = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_n_epochs


    # lr_bu
    algo = SVD(n_factors=1, n_epochs=1, lr_bu=5)
    rmse_lr_bu = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_lr_bu

    # lr_bi
    algo = SVD(n_factors=1, n_epochs=1, lr_bi=5)
    rmse_lr_bi = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_lr_bi

    # lr_pu
    algo = SVD(n_factors=1, n_epochs=1, lr_pu=5)
    rmse_lr_pu = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_lr_pu

    # lr_qi
    algo = SVD(n_factors=1, n_epochs=1, lr_qi=5)
    rmse_lr_qi = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_lr_qi

    # reg_bu
    algo = SVD(n_factors=1, n_epochs=1, reg_bu=5)
    rmse_reg_bu = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_reg_bu

    # reg_bi
    algo = SVD(n_factors=1, n_epochs=1, reg_bi=5)
    rmse_reg_bi = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_reg_bi

    # reg_pu
    algo = SVD(n_factors=1, n_epochs=1, reg_pu=5)
    rmse_reg_pu = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_reg_pu

    # reg_qi
    algo = SVD(n_factors=1, n_epochs=1, reg_qi=5)
    rmse_reg_qi = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_reg_qi
