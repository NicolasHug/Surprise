"""
Module for testing the Dataset class.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

import pytest

from surprise import Dataset
from surprise import Reader


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


def test_split():
    """Test the split method."""

    custom_dataset_path = (os.path.dirname(os.path.realpath(__file__)) +
                           '/custom_dataset')
    data = Dataset.load_from_file(file_path=custom_dataset_path, reader=reader)

    # Test n_folds parameter
    data.split(5)
    assert len(list(data.folds())) == 5

    with pytest.raises(ValueError):
        data.split(10)
        for fold in data.folds():
            pass

    with pytest.raises(ValueError):
        data.split(1)
        for fold in data.folds():
            pass

    # Test the shuffle parameter
    data.split(n_folds=3, shuffle=False)
    testsets_a = [testset for (_, testset) in data.folds()]
    data.split(n_folds=3, shuffle=False)
    testsets_b = [testset for (_, testset) in data.folds()]
    assert testsets_a == testsets_b

    # We'll shuffle and check that folds are now different. There's a chance
    # that they're still the same, just by lack of luck. If after 10000 tries
    # the're still the same, there's a high probability that our code is
    # faulty. If we're very (very very very) unlucky, it may fail though (or
    # loop for eternity).
    i = 0
    while testsets_a == testsets_b:
        data.split(n_folds=3, shuffle=True)
        testsets_b = [testset for (_, testset) in data.folds()]
        i += 1
    assert i < 10000

    # Ensure that folds are the same if split is not called again
    testsets_a = [testset for (_, testset) in data.folds()]
    testsets_b = [testset for (_, testset) in data.folds()]
    assert testsets_a == testsets_b


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

    # test raw2inner: ensure inner ids are given in proper order
    raw2inner_id_users = trainset._raw2inner_id_users
    for i in range(4):
        assert raw2inner_id_users['user' + str(i)] == i

    raw2inner_id_items = trainset._raw2inner_id_items
    for i in range(2):
        assert raw2inner_id_items['item' + str(i)] == i
