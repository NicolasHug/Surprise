"""
Module for testing the CoClustering algorithm.
"""


from surprise import CoClustering
from surprise.model_selection import cross_validate


def test_CoClustering_parameters(u1_ml100k, pkf):
    """Ensure that all parameters are taken into account."""

    # The baseline against which to compare.
    algo = CoClustering(n_epochs=1, random_state=1)
    rmse_default = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]

    # n_cltr_u
    algo = CoClustering(n_cltr_u=1, n_epochs=1, random_state=1)
    rmse_n_cltr_u = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_n_cltr_u

    # n_cltr_i
    algo = CoClustering(n_cltr_i=1, n_epochs=1, random_state=1)
    rmse_n_cltr_i = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_n_cltr_i

    # n_epochs
    algo = CoClustering(n_epochs=2, random_state=1)
    rmse_n_epochs = cross_validate(algo, u1_ml100k, ["rmse"], pkf)["test_rmse"]
    assert rmse_default != rmse_n_epochs
