"""Ensure that options for baseline estimates are taken into account."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

import pytest

from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise import evaluate


# the test and train files are from the ml-100k dataset (10% of u1.base and
# 10 % of u1.test)
train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
data = Dataset.load_from_folds([(train_file, test_file)], Reader('ml-100k'))


def test_method_field():
    """Ensure the method field is taken into account."""

    bsl_options = {'method': 'als'}
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als = evaluate(algo, data, measures=['rmse'])['rmse']

    bsl_options = {'method': 'sgd'}
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd = evaluate(algo, data, measures=['rmse'])['rmse']

    assert rmse_als != rmse_sgd

    with pytest.raises(ValueError):
        bsl_options = {'method': 'wrong_name'}
        algo = BaselineOnly(bsl_options=bsl_options)
        evaluate(algo, data)


def test_als_n_epochs_field():
    """Ensure the n_epochs field is taken into account."""

    bsl_options = {'method': 'als',
                   'n_epochs': 1,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_n_epochs_1 = evaluate(algo, data, measures=['rmse'])['rmse']

    bsl_options = {'method': 'als',
                   'n_epochs': 5,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_n_epochs_5 = evaluate(algo, data, measures=['rmse'])['rmse']

    assert rmse_als_n_epochs_1 != rmse_als_n_epochs_5


def test_als_reg_u_field():
    """Ensure the reg_u field is taken into account."""

    bsl_options = {'method': 'als',
                   'reg_u': 0,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regu_0 = evaluate(algo, data, measures=['rmse'])['rmse']

    bsl_options = {'method': 'als',
                   'reg_u': 10,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regu_10 = evaluate(algo, data, measures=['rmse'])['rmse']

    assert rmse_als_regu_0 != rmse_als_regu_10


def test_als_reg_i_field():
    """Ensure the reg_i field is taken into account."""

    bsl_options = {'method': 'als',
                   'reg_i': 0,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regi_0 = evaluate(algo, data, measures=['rmse'])['rmse']

    bsl_options = {'method': 'als',
                   'reg_i': 10,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regi_10 = evaluate(algo, data, measures=['rmse'])['rmse']

    assert rmse_als_regi_0 != rmse_als_regi_10


def test_sgd_n_epoch_field():
    """Ensure the n_epoch field is taken into account."""

    bsl_options = {'method': 'sgd',
                   'n_epochs': 1,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_n_epoch_1 = evaluate(algo, data, measures=['rmse'])['rmse']

    bsl_options = {'method': 'sgd',
                   'n_epochs': 20,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_n_epoch_5 = evaluate(algo, data, measures=['rmse'])['rmse']

    assert rmse_sgd_n_epoch_1 != rmse_sgd_n_epoch_5


def test_sgd_learning_rate_field():
    """Ensure the learning_rate field is taken into account."""

    bsl_options = {'method': 'sgd',
                   'n_epochs': 1,
                   'learning_rate': .005,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_lr_005 = evaluate(algo, data, measures=['rmse'])['rmse']

    bsl_options = {'method': 'sgd',
                   'n_epochs': 1,
                   'learning_rate': .00005,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_lr_00005 = evaluate(algo, data, measures=['rmse'])['rmse']

    assert rmse_sgd_lr_005 != rmse_sgd_lr_00005


def test_sgd_reg_field():
    """Ensure the reg field is taken into account."""

    bsl_options = {'method': 'sgd',
                   'n_epochs': 1,
                   'reg': 0.02,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_reg_002 = evaluate(algo, data, measures=['rmse'])['rmse']

    bsl_options = {'method': 'sgd',
                   'n_epochs': 1,
                   'reg': 1,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_reg_1 = evaluate(algo, data, measures=['rmse'])['rmse']

    assert rmse_sgd_reg_002 != rmse_sgd_reg_1
