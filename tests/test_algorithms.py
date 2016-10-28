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
    """Ensure that all algorithms act gracefully when asked to predict a rating
    of an unknown user, an unknown item, and when both are unknown.
    """

    reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                    rating_scale=(1, 5))

    file_path = os.path.dirname(os.path.realpath(__file__)) + '/custom_train'

    data = Dataset.load_from_file(file_path=file_path, reader=reader)

    for trainset, testset in data.folds:
        pass # just need trainset and testset to be set

    klasses = (NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans,
               KNNBaseline, SVD, SVDpp)
    for klass in klasses:
        print(klass)
        algo = klass()
        algo.train(trainset)
        algo.predict(0, 'unknown_item')
        algo.predict('unkown_user', 0)
        algo.predict('unkown_user', 'unknown_item')
