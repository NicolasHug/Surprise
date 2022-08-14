"""
Module for testing prediction algorithms.
"""



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
from surprise import KNNWithZScore
from surprise.model_selection import train_test_split
from surprise import accuracy


def test_unknown_user_or_item(toy_data):
    """Ensure that all algorithms act gracefully when asked to predict a rating
    of an unknown user, an unknown item, and when both are unknown.
    """

    trainset = toy_data.build_full_trainset()

    klasses = (NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans,
               KNNBaseline, SVD, SVDpp, NMF, SlopeOne, CoClustering,
               KNNWithZScore)
    for klass in klasses:
        algo = klass()
        algo.fit(trainset)
        algo.predict('user0', 'unknown_item', None)
        algo.predict('unknown_user', 'item0', None)
        algo.predict('unknown_user', 'unknown_item', None)

    # unrelated, but test the fit().test() one-liner:
    trainset, testset = train_test_split(toy_data, test_size=2)
    for klass in klasses:
        algo = klass()
        algo.fit(trainset).test(testset)


def test_knns(u1_ml100k, pkf):
    """Ensure the k and min_k parameters are effective for knn algorithms."""

    # Actually, as KNNWithMeans and KNNBaseline have back up solutions for when
    # there are not enough neighbors, we can't really test them...
    klasses = (KNNBasic, )  # KNNWithMeans, KNNBaseline)

    k, min_k = 20, 5
    for klass in klasses:
        algo = klass(k=k, min_k=min_k)
        for trainset, testset in pkf.split(u1_ml100k):
            algo.fit(trainset)
            predictions = algo.test(testset)
            for pred in predictions:
                if not pred.details['was_impossible']:
                    assert min_k <= pred.details['actual_k'] <= k


def test_nearest_neighbors():
    """Ensure the nearest neighbors are different when using user-user
    similarity vs item-item."""

    reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                    rating_scale=(1, 5))

    data_file = os.path.dirname(os.path.realpath(__file__)) + '/custom_train'
    data = Dataset.load_from_file(data_file, reader)
    trainset = data.build_full_trainset()

    algo_ub = KNNBasic(sim_options={'user_based': True})
    algo_ub.fit(trainset)
    algo_ib = KNNBasic(sim_options={'user_based': False})
    algo_ib.fit(trainset)
    assert algo_ub.get_neighbors(0, k=10) != algo_ib.get_neighbors(0, k=10)


def test_sanity_checks(u1_ml100k, pkf):
    """
    Basic sanity checks for all algorithms: check that RMSE stays the same.
    """

    expected_rmse = {
        BaselineOnly: 1.0268524031297395,
        KNNBasic: 1.1337265249554591,
        KNNWithMeans: 1.1043129441881696,
        KNNBaseline: 1.0700718041752253,
        KNNWithZScore: 1.11179436167853,
        SVD: 1.0077323320656948,
        SVDpp: 1.00284553561452,
        NMF: 1.0865370266372372,
        SlopeOne: 1.1559939123891685,
        CoClustering: 1.0841941385276614,
    }

    for klass, rmse in expected_rmse.items():
        if klass in (SVD, SVDpp, NMF, CoClustering):
            algo = klass(random_state=0)
        else:
            algo = klass()
        trainset, testset = next(pkf.split(u1_ml100k))
        algo.fit(trainset)
        predictions = algo.test(testset)
        assert accuracy.rmse(predictions, verbose=False) == rmse
