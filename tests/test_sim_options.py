"""
Module for testing the sim_options parameter.
"""


from itertools import combinations

import pytest

from surprise import KNNBaseline, KNNBasic, KNNWithMeans
from surprise.model_selection import cross_validate


def test_name_field(u1_ml100k, pkf):
    """Ensure the name field is taken into account."""

    sim_options = {"name": "cosine"}
    algo = KNNBasic(sim_options=sim_options)
    rmse_cosine = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    sim_options = {"name": "msd"}
    algo = KNNBasic(sim_options=sim_options)
    rmse_msd = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    sim_options = {"name": "pearson"}
    algo = KNNBasic(sim_options=sim_options)
    rmse_pearson = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    sim_options = {"name": "pearson_baseline"}
    bsl_options = {"n_epochs": 1}
    algo = KNNBasic(sim_options=sim_options, bsl_options=bsl_options)
    rmse_pearson_bsl = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    for rmse_a, rmse_b in combinations(
        (rmse_cosine, rmse_msd, rmse_pearson, rmse_pearson_bsl), 2
    ):
        assert rmse_a != rmse_b

    with pytest.raises(NameError):
        sim_options = {"name": "wrong_name"}
        algo = KNNBasic(sim_options=sim_options)
        cross_validate(algo, u1_ml100k, ["rmse"], pkf)


def test_user_based_field(u1_ml100k, pkf):
    """Ensure that the user_based field is taken into account (only) when
    needed."""

    algorithms = (KNNBasic, KNNWithMeans, KNNBaseline)
    for klass in algorithms:
        algo = klass(sim_options={"user_based": True})
        rmses_user_based = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
        algo = klass(sim_options={"user_based": False})
        rmses_item_based = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
        assert rmses_user_based != rmses_item_based


def test_shrinkage_field(u1_ml100k, pkf):
    """Ensure the shrinkage field is taken into account."""

    sim_options = {"name": "pearson_baseline", "shrinkage": 0}
    bsl_options = {"n_epochs": 1}
    algo = KNNBasic(sim_options=sim_options)
    rmse_shrinkage_0 = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    sim_options = {"name": "pearson_baseline", "shrinkage": 100}
    bsl_options = {"n_epochs": 1}
    algo = KNNBasic(sim_options=sim_options, bsl_options=bsl_options)
    rmse_shrinkage_100 = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    assert rmse_shrinkage_0 != rmse_shrinkage_100
