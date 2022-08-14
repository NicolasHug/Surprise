"""Ensure that options for baseline estimates are taken into account."""



import pytest

from surprise import BaselineOnly
from surprise.model_selection import cross_validate


def test_method_field(u1_ml100k, pkf):
    """Ensure the method field is taken into account."""

    bsl_options = {'method': 'als'}
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    bsl_options = {'method': 'sgd'}
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    assert rmse_als != rmse_sgd

    with pytest.raises(ValueError):
        bsl_options = {'method': 'wrong_name'}
        algo = BaselineOnly(bsl_options=bsl_options)
        cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']


def test_als_n_epochs_field(u1_ml100k, pkf):
    """Ensure the n_epochs field is taken into account."""

    bsl_options = {'method': 'als',
                   'n_epochs': 1,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_n_epochs_1 = cross_validate(algo, u1_ml100k, ['rmse'],
                                         pkf)['test_rmse']

    bsl_options = {'method': 'als',
                   'n_epochs': 5,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_n_epochs_5 = cross_validate(algo, u1_ml100k, ['rmse'],
                                         pkf)['test_rmse']

    assert rmse_als_n_epochs_1 != rmse_als_n_epochs_5


def test_als_reg_u_field(u1_ml100k, pkf):
    """Ensure the reg_u field is taken into account."""

    bsl_options = {'method': 'als',
                   'reg_u': 0,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regu_0 = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    bsl_options = {'method': 'als',
                   'reg_u': 10,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regu_10 = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    assert rmse_als_regu_0 != rmse_als_regu_10


def test_als_reg_i_field(u1_ml100k, pkf):
    """Ensure the reg_i field is taken into account."""

    bsl_options = {'method': 'als',
                   'reg_i': 0,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regi_0 = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    bsl_options = {'method': 'als',
                   'reg_i': 10,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regi_10 = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    assert rmse_als_regi_0 != rmse_als_regi_10


def test_sgd_n_epoch_field(u1_ml100k, pkf):
    """Ensure the n_epoch field is taken into account."""

    bsl_options = {'method': 'sgd',
                   'n_epochs': 1,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_n_epoch_1 = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    bsl_options = {'method': 'sgd',
                   'n_epochs': 20,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_n_epoch_5 = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    assert rmse_sgd_n_epoch_1 != rmse_sgd_n_epoch_5


def test_sgd_learning_rate_field(u1_ml100k, pkf):
    """Ensure the learning_rate field is taken into account."""

    bsl_options = {'method': 'sgd',
                   'n_epochs': 1,
                   'learning_rate': .005,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_lr_005 = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    bsl_options = {'method': 'sgd',
                   'n_epochs': 1,
                   'learning_rate': .00005,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_lr_00005 = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    assert rmse_sgd_lr_005 != rmse_sgd_lr_00005


def test_sgd_reg_field(u1_ml100k, pkf):
    """Ensure the reg field is taken into account."""

    bsl_options = {'method': 'sgd',
                   'n_epochs': 1,
                   'reg': 0.02,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_reg_002 = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    bsl_options = {'method': 'sgd',
                   'n_epochs': 1,
                   'reg': 1,
                   }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_reg_1 = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    assert rmse_sgd_reg_002 != rmse_sgd_reg_1
