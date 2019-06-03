"""
Module for testing prediction algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

import pytest
import pandas as pd

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
from surprise import KNNWithZScore
from surprise import Lasso
from surprise import FM
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import train_test_split


def test_unknown_user_or_item():
    """Ensure that all algorithms act gracefully when asked to predict a rating
    of an unknown user and/or an unknown item with unknown or known user
    features and/or unknown or known item features. Also, test how they react
    when features are missing/added for fit.
    """

    reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                    rating_scale=(1, 5))

    file_path = os.path.dirname(os.path.realpath(__file__)) + '/custom_dataset'

    # df with all users
    u_features_df = pd.DataFrame(
        {'urid': ['user0', 'user2', 'user3', 'user1', 'user4'],
         'isMale': [False, True, False, True, False]},
        columns=['urid', 'isMale'])

    # df with all items
    i_features_df = pd.DataFrame(
        {'irid': ['item0', 'item1'],
         'isNew': [False, True],
         'webRating': [4, 3],
         'isComedy': [True, False]},
        columns=['irid', 'isNew', 'webRating', 'isComedy'])

    # df with missing user
    u_features_m_df = pd.DataFrame(
        {'urid': ['user0', 'user2', 'user3', 'user1'],
         'isMale': [False, True, False, True]},
        columns=['urid', 'isMale'])

    # df with missing item
    i_features_m_df = pd.DataFrame(
        {'irid': ['item0'],
         'isNew': [False],
         'webRating': [4],
         'isComedy': [True]},
        columns=['irid', 'isNew', 'webRating', 'isComedy'])

    data = Dataset.load_from_file(file_path=file_path, reader=reader)
    trainset = data.build_full_trainset()

    data_u = Dataset.load_from_file(file_path=file_path, reader=reader)
    data_u.load_features_df(u_features_df, user_features=True)
    trainset_u = data_u.build_full_trainset()

    data_i = Dataset.load_from_file(file_path=file_path, reader=reader)
    data_i.load_features_df(i_features_df, user_features=False)
    trainset_i = data_i.build_full_trainset()

    data_ui = Dataset.load_from_file(file_path=file_path, reader=reader)
    data_ui.load_features_df(u_features_df, user_features=True)
    data_ui.load_features_df(i_features_df, user_features=False)
    trainset_ui = data_ui.build_full_trainset()

    data_ui_mu = Dataset.load_from_file(file_path=file_path, reader=reader)
    data_ui_mu.load_features_df(u_features_m_df, user_features=True)
    data_ui_mu.load_features_df(i_features_df, user_features=False)
    with pytest.raises(ValueError):
        data_ui_mu.build_full_trainset()

    data_ui_mi = Dataset.load_from_file(file_path=file_path, reader=reader)
    data_ui_mi.load_features_df(u_features_df, user_features=True)
    data_ui_mi.load_features_df(i_features_m_df, user_features=False)
    with pytest.raises(ValueError):
        data_ui_mi.build_full_trainset()

    # algos not using features
    klasses = (NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans,
               KNNBaseline, SVD, SVDpp, NMF, SlopeOne, CoClustering,
               KNNWithZScore)
    for klass in klasses:
        algo = klass()
        algo.fit(trainset)
        algo.fit(trainset_u)
        algo.fit(trainset_i)
        algo.fit(trainset_ui)
        algo.predict('user0', 'unknown_item')
        algo.predict('unkown_user', 'item0')
        algo.predict('unkown_user', 'unknown_item')
        algo.predict('user0', 'unknown_item', [], [])
        algo.predict('unkown_user', 'item0', [], [])
        algo.predict('unkown_user', 'unknown_item', [], [])
        algo.predict('user0', 'unknown_item', [False], [])
        algo.predict('unkown_user', 'item0', [False], [])
        algo.predict('unkown_user', 'unknown_item', [False], [])
        algo.predict('user0', 'unknown_item', [], [False, 4, True])
        algo.predict('unkown_user', 'item0', [], [False, 4, True])
        algo.predict('unkown_user', 'unknown_item', [], [False, 4, True])
        algo.predict('user0', 'unknown_item', [False], [False, 4, True])
        algo.predict('unkown_user', 'item0', [False], [False, 4, True])
        algo.predict('unkown_user', 'unknown_item', [False], [False, 4, True])

    # algos using user and item features
    klasses_ui = (Lasso,)
    for klass in klasses_ui:
        algo = klass()
        with pytest.raises(ValueError):
            algo.fit(trainset)
        with pytest.raises(ValueError):
            algo.fit(trainset_u)
        with pytest.raises(ValueError):
            algo.fit(trainset_i)
        algo.fit(trainset_ui)
        algo.predict('user0', 'unknown_item')
        algo.predict('unkown_user', 'item0')
        algo.predict('unkown_user', 'unknown_item')
        algo.predict('user0', 'unknown_item', [], [])
        algo.predict('unkown_user', 'item0', [], [])
        algo.predict('unkown_user', 'unknown_item', [], [])
        algo.predict('user0', 'unknown_item', [False], [])
        algo.predict('unkown_user', 'item0', [False], [])
        algo.predict('unkown_user', 'unknown_item', [False], [])
        algo.predict('user0', 'unknown_item', [], [False, 4, True])
        algo.predict('unkown_user', 'item0', [], [False, 4, True])
        algo.predict('unkown_user', 'unknown_item', [], [False, 4, True])
        algo.predict('user0', 'unknown_item', [False], [False, 4, True])
        algo.predict('unkown_user', 'item0', [False], [False, 4, True])
        algo.predict('unkown_user', 'unknown_item', [False], [False, 4, True])

    # FM using (optionally) user and item features
    rating_lst_opt = ['userID', 'itemID', 'imp_u_rating', 'exp_u_rating']
    user_lst_opt = ['isMale']
    item_lst_opt = ['isNew', 'webRating', 'isComedy']
    algos = (FM(rating_lst=rating_lst_opt), FM(user_lst=user_lst_opt),
             FM(item_lst=item_lst_opt),
             FM(rating_lst=rating_lst_opt, user_lst=user_lst_opt,
                item_lst=item_lst_opt))
    for algo in algos:
        if algo.user_lst:
            with pytest.raises(ValueError):
                algo.fit(trainset)
            with pytest.raises(ValueError):
                algo.fit(trainset_i)
        if algo.item_lst:
            with pytest.raises(ValueError):
                algo.fit(trainset)
            with pytest.raises(ValueError):
                algo.fit(trainset_u)
        algo.fit(trainset_ui)
        if algo.user_lst:
            with pytest.raises(ValueError):
                algo.predict('user0', 'item0', [False, 'AddedFeature'], [])
        if algo.item_lst:
            with pytest.raises(ValueError):
                algo.predict('user0', 'item0', [], ['NotEnoughFeatures'])
        algo.predict('user0', 'unknown_item')
        algo.predict('unkown_user', 'item0')
        algo.predict('unkown_user', 'unknown_item')
        algo.predict('user0', 'unknown_item', [], [])
        algo.predict('unkown_user', 'item0', [], [])
        algo.predict('unkown_user', 'unknown_item', [], [])
        algo.predict('user0', 'unknown_item', [False], [])
        algo.predict('unkown_user', 'item0', [False], [])
        algo.predict('unkown_user', 'unknown_item', [False], [])
        algo.predict('user0', 'unknown_item', [], [False, 4, True])
        algo.predict('unkown_user', 'item0', [], [False, 4, True])
        algo.predict('unkown_user', 'unknown_item', [], [False, 4, True])
        algo.predict('user0', 'unknown_item', [False], [False, 4, True])
        algo.predict('unkown_user', 'item0', [False], [False, 4, True])
        algo.predict('unkown_user', 'unknown_item', [False], [False, 4, True])

    # unrelated, but test the fit().test() one-liner:
    trainset, testset = train_test_split(data_ui, test_size=2)
    for klass in (klasses + klasses_ui + (FM,)):
        algo = klass()
        algo.fit(trainset).test(testset)
        with pytest.warns(UserWarning):
            algo.train(trainset).test(testset)


def test_knns():
    """Ensure the k and min_k parameters are effective for knn algorithms."""

    # the test and train files are from the ml-100k dataset (10% of u1.base and
    # 10 % of u1.test)
    train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
    test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
    data = Dataset.load_from_folds([(train_file, test_file)],
                                   Reader('ml-100k'))
    pkf = PredefinedKFold()

    # Actually, as KNNWithMeans and KNNBaseline have back up solutions for when
    # there are not enough neighbors, we can't really test them...
    klasses = (KNNBasic, )  # KNNWithMeans, KNNBaseline)

    k, min_k = 20, 5
    for klass in klasses:
        algo = klass(k=k, min_k=min_k)
        for trainset, testset in pkf.split(data):
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
