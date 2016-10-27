"""
Module for testing prediction algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np

from recsys.prediction_algorithms import *
from recsys.dataset import Dataset
from recsys.dataset import Reader
from recsys.evaluate import evaluate


def test_unknown_user_or_item():
    """Ensure that an unknown user or item in testset will predict the mean
    rating and that was_impossible is set to True."""

    reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                    rating_scale=(1, 5))

    current_dir = os.path.dirname(os.path.realpath(__file__))
    folds_files = [(current_dir + '/custom_train',
                    current_dir + '/custom_test')]

    data = Dataset.load_from_folds(folds_files=folds_files, reader=reader)

    for trainset, testset in data.folds:
        pass # just need trainset and testset to be set

    algo = NormalPredictor()
    algo.train(trainset)

    # set verbose to true for more coverage.
    predictions = algo.test(testset, verbose=True)

    global_mean = np.mean([r for (_, _, r) in algo.all_ratings])
    assert predictions[2].est == global_mean
    assert predictions[2].details['was_impossible'] == True
