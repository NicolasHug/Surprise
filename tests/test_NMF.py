"""
Module for testing the NMF algorithm.
"""


import pandas as pd
import pytest

from surprise import Dataset, NMF, Reader
from surprise.model_selection import cross_validate


def test_NMF_parameters(u1_ml100k, pkf):
    """Ensure that all parameters are taken into account."""

    # The baseline against which to compare.
    algo = NMF(n_factors=1, n_epochs=1, random_state=1)
    rmse_default = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    # n_factors
    algo = NMF(n_factors=2, n_epochs=1, random_state=1)
    rmse_factors = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_factors

    # n_epochs
    algo = NMF(n_factors=1, n_epochs=2, random_state=1)
    rmse_n_epochs = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_n_epochs

    # biased
    algo = NMF(n_factors=1, n_epochs=1, biased=True, random_state=1)
    rmse_biased = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_biased

    # reg_pu
    algo = NMF(n_factors=1, n_epochs=1, reg_pu=1, random_state=1)
    rmse_reg_pu = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_reg_pu

    # reg_qi
    algo = NMF(n_factors=1, n_epochs=1, reg_qi=1, random_state=1)
    rmse_reg_qi = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_reg_qi

    # reg_bu
    algo = NMF(n_factors=1, n_epochs=1, reg_bu=1, biased=True, random_state=1)
    rmse_reg_bu = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_reg_bu

    # reg_bi
    algo = NMF(n_factors=1, n_epochs=1, reg_bi=1, biased=True, random_state=1)
    rmse_reg_bi = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_reg_bi

    # lr_bu
    algo = NMF(n_factors=1, n_epochs=1, lr_bu=1, biased=True, random_state=1)
    rmse_lr_bu = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_lr_bu

    # lr_bi
    algo = NMF(n_factors=1, n_epochs=1, lr_bi=1, biased=True, random_state=1)
    rmse_lr_bi = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_lr_bi

    # init_low
    algo = NMF(n_factors=1, n_epochs=1, init_low=0.5, random_state=1)
    rmse_init_low = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_init_low

    # init_low
    with pytest.raises(ValueError):
        algo = NMF(n_factors=1, n_epochs=1, init_low=-1, random_state=1)

    # init_high
    algo = NMF(n_factors=1, n_epochs=1, init_high=0.5, random_state=1)
    rmse_init_high = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_init_high


def test_NMF_zero_ratings():
    # Non-regression test for https://github.com/NicolasHug/Surprise/pull/367
    reader = Reader(rating_scale=(-10, 10))

    ratings_dict = {
        "itemID": [0, 0, 0, 0, 1, 1],
        "userID": [0, 1, 2, 3, 3, 4],
        "rating": [-10, 10, 0, -5, 0, 5],
    }
    df = pd.DataFrame(ratings_dict)
    data = Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)
    trainset = data.build_full_trainset()
    algo = NMF(n_factors=4, n_epochs=2)
    algo.fit(trainset)
