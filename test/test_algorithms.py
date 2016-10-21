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

def test_baseline_computation():
    """Ensure that options for baseline estimates are taken into account."""

    # method
    bsl_options = {'method' : 'als'}
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als, _, _ = evaluate(algo, data)

    bsl_options = {'method' : 'sgd'}
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd , _, _ = evaluate(algo, data)

    assert rmse_als != rmse_sgd

    # als n_epochs
    bsl_options = {'method' : 'als',
                   'n_epochs' : 1,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_n_epochs_1, _, _ = evaluate(algo, data)

    bsl_options = {'method' : 'als',
                   'n_epochs' : 5,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_n_epochs_5, _, _ = evaluate(algo, data)

    assert rmse_als_n_epochs_1 != rmse_als_n_epochs_5

    # als reg_u
    bsl_options = {'method' : 'als',
                   'reg_u' : 0,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regu_0, _, _ = evaluate(algo, data)

    bsl_options = {'method' : 'als',
                   'reg_u' : 10,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regu_10, _, _ = evaluate(algo, data)

    assert rmse_als_regu_0!= rmse_als_regu_10

    # als reg_i
    bsl_options = {'method' : 'als',
                   'reg_i' : 0,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regi_0, _, _ = evaluate(algo, data)

    bsl_options = {'method' : 'als',
                   'reg_i' : 10,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_als_regi_10, _, _ = evaluate(algo, data)

    assert rmse_als_regi_0!= rmse_als_regi_10


    # sgd n_epoch
    bsl_options = {'method' : 'sgd',
                   'n_epochs' : 1,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_n_epoch_1, _, _ = evaluate(algo, data)

    bsl_options = {'method' : 'sgd',
                   'n_epochs' : 20,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_n_epoch_5, _, _ = evaluate(algo, data)

    assert rmse_sgd_n_epoch_1 != rmse_sgd_n_epoch_5

    # sgd learning_rate
    bsl_options = {'method' : 'sgd',
                   'n_epochs' : 1,
                   'learning_rate' : .005,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_lr_005, _, _ = evaluate(algo, data)

    bsl_options = {'method' : 'sgd',
                   'n_epochs' : 1,
                   'learning_rate' : .00005,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_lr_00005, _, _ = evaluate(algo, data)

    assert rmse_sgd_lr_005 != rmse_sgd_lr_00005

    # sgd reg
    bsl_options = {'method' : 'sgd',
                   'n_epochs' : 1,
                   'reg' : 0.02,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_reg_002, _, _ = evaluate(algo, data)

    bsl_options = {'method' : 'sgd',
                   'n_epochs' : 1,
                   'reg' : 1,
    }
    algo = BaselineOnly(bsl_options=bsl_options)
    rmse_sgd_reg_1, _, _ = evaluate(algo, data)

    assert rmse_sgd_reg_002 != rmse_sgd_reg_1
