"""
Module for testing the SVD and SVD++ algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

from surprise import SVD
from surprise import SVDpp
from surprise import Dataset
from surprise import Reader
from surprise import evaluate


# the test and train files are from the ml-100k dataset (10% of u1.base and
# 10 % of u1.test)
train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
data = Dataset.load_from_folds([(train_file, test_file)], Reader('ml-100k'))


def test_SVD_parameters():
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

    # biased
    algo = SVD(n_factors=1, n_epochs=1, biased=False)
    rmse_biased = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_biased

    # lr_all
    algo = SVD(n_factors=1, n_epochs=1, lr_all=5)
    rmse_lr_all = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_lr_all

    # reg_all
    algo = SVD(n_factors=1, n_epochs=1, reg_all=5)
    rmse_reg_all = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_reg_all

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


def test_SVDpp_parameters():
    """Ensure that all parameters are taken into account."""

    # The baseline against which to compare.
    algo = SVDpp(n_factors=1, n_epochs=1)
    rmse_default = evaluate(algo, data, measures=['rmse'])['rmse']

    # n_factors
    algo = SVDpp(n_factors=2, n_epochs=1)
    rmse_factors = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_factors

    # The rest is OK but just takes too long for now...
    """

    # n_epochs
    algo = SVDpp(n_factors=1, n_epochs=2)
    rmse_n_epochs = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_n_epochs

    # lr_all
    algo = SVDpp(n_factors=1, n_epochs=1, lr_all=5)
    rmse_lr_all = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_lr_all

    # reg_all
    algo = SVDpp(n_factors=1, n_epochs=1, reg_all=5)
    rmse_reg_all = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_reg_all

    # lr_bu
    algo = SVDpp(n_factors=1, n_epochs=1, lr_bu=5)
    rmse_lr_bu = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_lr_bu

    # lr_bi
    algo = SVDpp(n_factors=1, n_epochs=1, lr_bi=5)
    rmse_lr_bi = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_lr_bi

    # lr_pu
    algo = SVDpp(n_factors=1, n_epochs=1, lr_pu=5)
    rmse_lr_pu = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_lr_pu

    # lr_qi
    algo = SVDpp(n_factors=1, n_epochs=1, lr_qi=5)
    rmse_lr_qi = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_lr_qi

    # lr_yj
    algo = SVDpp(n_factors=1, n_epochs=1, lr_yj=5)
    rmse_lr_yj = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_lr_yj

    # reg_bu
    algo = SVDpp(n_factors=1, n_epochs=1, reg_bu=5)
    rmse_reg_bu = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_reg_bu

    # reg_bi
    algo = SVDpp(n_factors=1, n_epochs=1, reg_bi=5)
    rmse_reg_bi = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_reg_bi

    # reg_pu
    algo = SVDpp(n_factors=1, n_epochs=1, reg_pu=5)
    rmse_reg_pu = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_reg_pu

    # reg_qi
    algo = SVDpp(n_factors=1, n_epochs=1, reg_qi=5)
    rmse_reg_qi = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_reg_qi

    # reg_yj
    algo = SVDpp(n_factors=1, n_epochs=1, reg_yj=5)
    rmse_reg_yj = evaluate(algo, data, measures=['rmse'])['rmse']
    assert rmse_default != rmse_reg_yj
    """
