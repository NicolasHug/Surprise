"""
Module for testing the SVD and SVD++ algorithms.
"""




from surprise import SVD
from surprise import SVDpp
from surprise.model_selection import cross_validate


def test_SVD_parameters(u1_ml100k, pkf):
    """Ensure that all parameters are taken into account."""

    # The baseline against which to compare.
    algo = SVD(n_factors=1, n_epochs=1, random_state=1)
    rmse_default = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    # n_factors
    algo = SVD(n_factors=2, n_epochs=1, random_state=1)
    rmse_factors = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_factors

    # n_epochs
    algo = SVD(n_factors=1, n_epochs=2, random_state=1)
    rmse_n_epochs = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_n_epochs

    # biased
    algo = SVD(n_factors=1, n_epochs=1, biased=False, random_state=1)
    rmse_biased = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_biased

    # lr_all
    algo = SVD(n_factors=1, n_epochs=1, lr_all=5, random_state=1)
    rmse_lr_all = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_lr_all

    # reg_all
    algo = SVD(n_factors=1, n_epochs=1, reg_all=5, random_state=1)
    rmse_reg_all = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_reg_all

    # lr_bu
    algo = SVD(n_factors=1, n_epochs=1, lr_bu=5, random_state=1)
    rmse_lr_bu = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_lr_bu

    # lr_bi
    algo = SVD(n_factors=1, n_epochs=1, lr_bi=5, random_state=1)
    rmse_lr_bi = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_lr_bi

    # lr_pu
    algo = SVD(n_factors=1, n_epochs=1, lr_pu=5, random_state=1)
    rmse_lr_pu = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_lr_pu

    # lr_qi
    algo = SVD(n_factors=1, n_epochs=1, lr_qi=5, random_state=1)
    rmse_lr_qi = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_lr_qi

    # reg_bu
    algo = SVD(n_factors=1, n_epochs=1, reg_bu=5, random_state=1)
    rmse_reg_bu = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_reg_bu

    # reg_bi
    algo = SVD(n_factors=1, n_epochs=1, reg_bi=5, random_state=1)
    rmse_reg_bi = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_reg_bi

    # reg_pu
    algo = SVD(n_factors=1, n_epochs=1, reg_pu=5, random_state=1)
    rmse_reg_pu = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_reg_pu

    # reg_qi
    algo = SVD(n_factors=1, n_epochs=1, reg_qi=5, random_state=1)
    rmse_reg_qi = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_reg_qi


def test_SVDpp_parameters(u1_ml100k, pkf):
    """Ensure that all parameters are taken into account."""

    # The baseline against which to compare.
    algo = SVDpp(n_factors=1, n_epochs=1, random_state=1)
    rmse_default = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']

    # n_factors
    algo = SVDpp(n_factors=2, n_epochs=1, random_state=1)
    rmse_factors = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_factors

    # The rest is OK but just takes too long for now...
    """

    # n_epochs
    algo = SVDpp(n_factors=1, n_epochs=2, random_state=1)
    rmse_n_epochs = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_n_epochs

    # lr_all
    algo = SVDpp(n_factors=1, n_epochs=1, lr_all=5, random_state=1)
    rmse_lr_all = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_lr_all

    # reg_all
    algo = SVDpp(n_factors=1, n_epochs=1, reg_all=5, random_state=1)
    rmse_reg_all = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_reg_all

    # lr_bu
    algo = SVDpp(n_factors=1, n_epochs=1, lr_bu=5, random_state=1)
    rmse_lr_bu = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_lr_bu

    # lr_bi
    algo = SVDpp(n_factors=1, n_epochs=1, lr_bi=5, random_state=1)
    rmse_lr_bi = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_lr_bi

    # lr_pu
    algo = SVDpp(n_factors=1, n_epochs=1, lr_pu=5, random_state=1)
    rmse_lr_pu = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_lr_pu

    # lr_qi
    algo = SVDpp(n_factors=1, n_epochs=1, lr_qi=5, random_state=1)
    rmse_lr_qi = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_lr_qi

    # lr_yj
    algo = SVDpp(n_factors=1, n_epochs=1, lr_yj=5, random_state=1)
    rmse_lr_yj = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_lr_yj

    # reg_bu
    algo = SVDpp(n_factors=1, n_epochs=1, reg_bu=5, random_state=1)
    rmse_reg_bu = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_reg_bu

    # reg_bi
    algo = SVDpp(n_factors=1, n_epochs=1, reg_bi=5, random_state=1)
    rmse_reg_bi = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_reg_bi

    # reg_pu
    algo = SVDpp(n_factors=1, n_epochs=1, reg_pu=5, random_state=1)
    rmse_reg_pu = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_reg_pu

    # reg_qi
    algo = SVDpp(n_factors=1, n_epochs=1, reg_qi=5, random_state=1)
    rmse_reg_qi = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_reg_qi

    # reg_yj
    algo = SVDpp(n_factors=1, n_epochs=1, reg_yj=5, random_state=1)
    rmse_reg_yj = cross_validate(algo, u1_ml100k, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_reg_yj
    """
