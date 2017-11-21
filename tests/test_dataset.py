"""
Module for testing the Dataset class.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import random

import pytest
import pandas as pd

from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader


random.seed(1)
reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                rating_scale=(1, 5))


def test_wrong_file_name():
    """Ensure file names are checked when creating a (custom) Dataset."""
    wrong_files = [('does_not_exist', 'does_not_either')]

    with pytest.raises(ValueError):
        Dataset.load_from_folds(folds_files=wrong_files, reader=reader)


def test_build_full_trainset():
    """Test the build_full_trainset method."""

    custom_dataset_path = (os.path.dirname(os.path.realpath(__file__)) +
                           '/custom_dataset')
    data = Dataset.load_from_file(file_path=custom_dataset_path, reader=reader)

    trainset = data.build_full_trainset()

    assert len(trainset.ur) == 5
    assert len(trainset.ir) == 2
    assert trainset.n_users == 5
    assert trainset.n_items == 2


def test_no_call_to_split():
    """Ensure, as mentioned in the split() docstring, that even if split is not
    called then the data is split with 5 folds after being shuffled."""

    custom_dataset_path = (os.path.dirname(os.path.realpath(__file__)) +
                           '/custom_dataset')
    data = Dataset.load_from_file(file_path=custom_dataset_path, reader=reader)

    assert len(list(data.folds())) == 5

    # make sure data has been shuffled. If not shuffled, the users in the
    # testsets would be 0, 1, 2... 4 (in that order).
    users = [int(testset[0][0][-1]) for (_, testset) in data.folds()]
    assert users != list(range(5))


def test_split():
    """Test the split method."""

    custom_dataset_path = (os.path.dirname(os.path.realpath(__file__)) +
                           '/custom_dataset')
    data = Dataset.load_from_file(file_path=custom_dataset_path, reader=reader)

    # Test the shuffle parameter
    # Make sure data has not been shuffled. If not shuffled, the users in the
    # testsets are be 0, 1, 2... 4 (in that order).
    data.split(n_folds=5, shuffle=False)
    users = [int(testset[0][0][-1]) for (_, testset) in data.folds()]
    assert users == list(range(5))

    # Test the shuffle parameter
    # Make sure that when called two times without shuffling, folds are the
    # same.
    data.split(n_folds=3, shuffle=False)
    testsets_a = [testset for (_, testset) in data.folds()]
    data.split(n_folds=3, shuffle=False)
    testsets_b = [testset for (_, testset) in data.folds()]
    assert testsets_a == testsets_b

    # We'll now shuffle b and check that folds are different.
    data.split(n_folds=3, shuffle=True)
    testsets_b = [testset for (_, testset) in data.folds()]
    assert testsets_a != testsets_b

    # Ensure that folds are the same if split is not called again
    testsets_a = [testset for (_, testset) in data.folds()]
    testsets_b = [testset for (_, testset) in data.folds()]
    assert testsets_a == testsets_b

    # Test n_folds parameter
    data.split(5)
    assert len(list(data.folds())) == 5

    with pytest.raises(ValueError):
        data.split(10)  # Too big (greater than number of ratings)

    with pytest.raises(ValueError):
        data.split(1)  # Too low (must be >= 2)


def test_trainset_testset():
    """Test the construct_trainset and construct_testset methods."""

    current_dir = os.path.dirname(os.path.realpath(__file__))
    folds_files = [(current_dir + '/custom_train',
                    current_dir + '/custom_test')]

    data = Dataset.load_from_folds(folds_files=folds_files, reader=reader)

    for trainset, testset in data.folds():
        pass  # just need trainset and testset to be set

    # test ur
    ur = trainset.ur
    assert ur[0] == [(0, 4)]
    assert ur[1] == [(0, 4), (1, 2)]
    assert ur[40] == []  # not in the trainset

    # test ir
    ir = trainset.ir
    assert ir[0] == [(0, 4), (1, 4), (2, 1)]
    assert ir[1] == [(1, 2), (2, 1), (3, 5)]
    assert ir[20000] == []  # not in the trainset

    # test n_users, n_items, n_ratings, rating_scale
    assert trainset.n_users == 4
    assert trainset.n_items == 2
    assert trainset.n_ratings == 6
    assert trainset.rating_scale == (1, 5)

    # test raw2inner
    for i in range(4):
        assert trainset.to_inner_uid('user' + str(i)) == i
    with pytest.raises(ValueError):
        trainset.to_inner_uid('unkown_user')

    for i in range(2):
        assert trainset.to_inner_iid('item' + str(i)) == i
    with pytest.raises(ValueError):
        trainset.to_inner_iid('unkown_item')

    # test inner2raw
    assert trainset._inner2raw_id_users is None
    assert trainset._inner2raw_id_items is None
    for i in range(4):
        assert trainset.to_raw_uid(i) == 'user' + str(i)
    for i in range(2):
        assert trainset.to_raw_iid(i) == 'item' + str(i)
    assert trainset._inner2raw_id_users is not None
    assert trainset._inner2raw_id_items is not None

    # Test the build_testset() method
    algo = BaselineOnly()
    algo.train(trainset)
    testset = trainset.build_testset()
    algo.test(testset)  # ensure an algorithm can manage the data
    assert ('user0', 'item0', 4) in testset
    assert ('user3', 'item1', 5) in testset
    assert ('user3', 'item1', 0) not in testset

    # Test the build_anti_testset() method
    algo = BaselineOnly()
    algo.train(trainset)
    testset = trainset.build_anti_testset()
    algo.test(testset)  # ensure an algorithm can manage the data
    assert ('user0', 'item0', trainset.global_mean) not in testset
    assert ('user3', 'item1', trainset.global_mean) not in testset
    assert ('user0', 'item1', trainset.global_mean) in testset
    assert ('user3', 'item0', trainset.global_mean) in testset


def test_load_form_df():
    """Ensure reading dataset from pandas dataframe is OK."""

    # DF creation.
    ratings_dict = {'itemID': [1, 1, 1, 2, 2],
                    'userID': [9, 32, 2, 45, '10000'],
                    'rating': [3, 2, 4, 3, 1]}
    df = pd.DataFrame(ratings_dict)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

    # Assert split and folds can be used without problems
    data.split(2)
    assert sum(1 for _ in data.folds()) == 2

    # assert users and items are correctly mapped
    trainset = data.build_full_trainset()
    assert trainset.knows_user(trainset.to_inner_uid(9))
    assert trainset.knows_user(trainset.to_inner_uid('10000'))
    assert trainset.knows_item(trainset.to_inner_iid(2))

    # assert r(9, 1) = 3 and r(2, 1) = 4
    uid9 = trainset.to_inner_uid(9)
    uid2 = trainset.to_inner_uid(2)
    iid1 = trainset.to_inner_iid(1)
    assert trainset.ur[uid9] == [(iid1, 3)]
    assert trainset.ur[uid2] == [(iid1, 4)]

    # assert at least rating file or dataframe must be specified
    with pytest.raises(ValueError):
        data = Dataset.load_from_df(None, None)

    # mess up the column ordering and assert that users are not correctly
    # mapped
    data = Dataset.load_from_df(df[['rating', 'itemID', 'userID']], reader)
    trainset = data.build_full_trainset()
    with pytest.raises(ValueError):
        trainset.to_inner_uid('10000')


def test_build_anti_testset():
    ratings_dict = {'itemID': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'userID': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    'rating': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
    df = pd.DataFrame(ratings_dict)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    data.split(2)
    trainset, __testset = next(data.folds())
    # fill with some specific value
    for fillvalue in (0, 42., -1):
        anti = trainset.build_anti_testset(fill=fillvalue)
        for (u, i, r) in anti:
            assert r == fillvalue
    # fill with global_mean
    anti = trainset.build_anti_testset(fill=None)
    for (u, i, r) in anti:
        assert r == trainset.global_mean
    expect = trainset.n_users * trainset.n_items
    assert trainset.n_ratings + len(anti) == expect
