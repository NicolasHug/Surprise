"""
Module for testing the CoClustering algorithm.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

from surprise import CoClustering
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import PredefinedKFold

# the test and train files are from the ml-100k dataset (10% of u1.base and
# 10 % of u1.test)
train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
data = Dataset.load_from_folds([(train_file, test_file)], Reader('ml-100k'))
pkf = PredefinedKFold()


def test_CoClustering_parameters():
    """Ensure that all parameters are taken into account."""

    # The baseline against which to compare.
    algo = CoClustering(n_epochs=1, random_state=1)
    rmse_default = cross_validate(algo, data, ['rmse'], pkf)['test_rmse']

    # n_cltr_u
    algo = CoClustering(n_cltr_u=1, n_epochs=1, random_state=1)
    rmse_n_cltr_u = cross_validate(algo, data, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_n_cltr_u

    # n_cltr_i
    algo = CoClustering(n_cltr_i=1, n_epochs=1, random_state=1)
    rmse_n_cltr_i = cross_validate(algo, data, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_n_cltr_i

    # n_epochs
    algo = CoClustering(n_epochs=2, random_state=1)
    rmse_n_epochs = cross_validate(algo, data, ['rmse'], pkf)['test_rmse']
    assert rmse_default != rmse_n_epochs
