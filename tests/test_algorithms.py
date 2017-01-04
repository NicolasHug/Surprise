"""
Module for testing prediction algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise import Dataset
from surprise import Reader


def test_unknown_user_or_item():
    """Ensure that all algorithms act gracefully when asked to predict a rating
    of an unknown user, an unknown item, and when both are unknown.
    """

    reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                    rating_scale=(1, 5))

    file_path = os.path.dirname(os.path.realpath(__file__)) + '/custom_train'

    data = Dataset.load_from_file(file_path=file_path, reader=reader)

    for trainset, testset in data.folds():
        pass  # just need trainset and testset to be set

    klasses = (NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans,
               KNNBaseline, SVD, SVDpp, NMF, SlopeOne, CoClustering)
    for klass in klasses:
        algo = klass()
        algo.train(trainset)
        algo.predict(0, 'unknown_item', None)
        algo.predict('unkown_user', 0, None)
        algo.predict('unkown_user', 'unknown_item', None)


def test_knns():
    """Ensure the k and min_k parameters are effective for knn algorithms."""

    # the test and train files are from the ml-100k dataset (10% of u1.base and
    # 10 % of u1.test)
    train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
    test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
    data = Dataset.load_from_folds([(train_file, test_file)],
                                   Reader('ml-100k'))

    # Actually, as KNNWithMeans and KNNBaseline have back up solutions for when
    # there are not enough neighbors, we can't really test them...
    klasses = (KNNBasic, )  # KNNWithMeans, KNNBaseline)

    k, min_k = 20, 5
    for klass in klasses:
        algo = klass(k=k, min_k=min_k)
        for trainset, testset in data.folds():
            algo.train(trainset)
            predictions = algo.test(testset)
            for pred in predictions:
                if not pred.details['was_impossible']:
                    assert min_k <= pred.details['actual_k'] <= k
