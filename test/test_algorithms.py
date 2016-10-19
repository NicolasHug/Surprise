"""
Module for testing prediction algorithms.
"""

import os
import numpy as np

from pyrec.prediction_algorithms import *
from pyrec.dataset import Dataset
from pyrec.dataset import Reader
from pyrec.evaluate import evaluate



# the test and train files are from the ml-100k dataset (10% of u1.base and
# 10 % of u1.test)
train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
data = Dataset.load_from_folds([(train_file, test_file)], Reader('ml-100k'))

def test_normal_predictor():
    """Just ensure that this algorithm runs gracefully without errors."""

    algo = NormalPredictor()
    evaluate(algo, data)

def test_user_based_param():
    """Ensure that the user_based parameter is taken into account (only) when
    needed."""

    algorithms = (KNNBasic, KNNWithMeans, KNNBaseline)
    for klass in algorithms:
        algo = klass(sim_options={'user_based':True})
        rmses_user_based, _, _ = evaluate(algo, data)
        algo = klass(sim_options={'user_based':False})
        rmses_item_based, _, _ = evaluate(algo, data)
        assert rmses_user_based != rmses_item_based

    algorithms = (BaselineOnly, )
    for klass in algorithms:
        algo = klass(sim_options={'user_based':True})
        rmses_user_based, _, _ = evaluate(algo, data)
        algo = klass(sim_options={'user_based':False})
        rmses_item_based, _, _ = evaluate(algo, data)
        assert np.allclose(rmses_user_based, rmses_item_based)
