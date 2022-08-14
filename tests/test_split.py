

import os
from copy import copy
import numpy as np
from collections import Counter

import pytest

from surprise import Dataset
from surprise import Reader
from surprise.model_selection import KFold
from surprise.model_selection import ShuffleSplit
from surprise.model_selection import RepeatedKFold
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise.model_selection import PredefinedKFold
from surprise.model_selection.split import get_cv


np.random.seed(1)


def test_KFold(toy_data):

    # Test n_folds parameter
    kf = KFold(n_splits=5)
    assert len(list(kf.split(toy_data))) == 5

    with pytest.raises(ValueError):
        kf = KFold(n_splits=10)
        next(kf.split(toy_data))  # Too big (greater than number of ratings)

    with pytest.raises(ValueError):
        kf = KFold(n_splits=1)
        next(kf.split(toy_data))  # Too low (must be >= 2)

    # Make sure data has not been shuffled. If not shuffled, the users in the
    # testsets are 0, 1, 2... 4 (in that order).
    kf = KFold(n_splits=5, shuffle=False)
    users = [int(testset[0][0][-1]) for (_, testset) in kf.split(toy_data)]
    assert users == list(range(5))

    # Make sure that when called two times without shuffling, folds are the
    # same.
    kf = KFold(n_splits=5, shuffle=False)
    testsets_a = [testset for (_, testset) in kf.split(toy_data)]
    testsets_b = [testset for (_, testset) in kf.split(toy_data)]
    assert testsets_a == testsets_b
    # test once again with another KFold instance
    kf = KFold(n_splits=5, shuffle=False)
    testsets_a = [testset for (_, testset) in kf.split(toy_data)]
    assert testsets_a == testsets_b

    # We'll now shuffle b and check that folds are different.
    # (this is conditioned by seed setting at the beginning of file)
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    testsets_b = [testset for (_, testset) in kf.split(toy_data)]
    assert testsets_a != testsets_b
    # test once again: two calls to kf.split make different splits when
    # random_state=None
    testsets_a = [testset for (_, testset) in kf.split(toy_data)]
    assert testsets_a != testsets_b

    # Make sure that folds are the same when same KFold instance is used with
    # suffle is True but random_state is set to some value
    kf = KFold(n_splits=5, random_state=1, shuffle=True)
    testsets_a = [testset for (_, testset) in kf.split(toy_data)]
    testsets_b = [testset for (_, testset) in kf.split(toy_data)]
    assert testsets_a == testsets_b

    # Make sure raw ratings are not shuffled by KFold
    old_raw_ratings = copy(toy_data.raw_ratings)
    kf = KFold(n_splits=5, shuffle=True)
    next(kf.split(toy_data))
    assert old_raw_ratings == toy_data.raw_ratings


def test_ShuffleSplit(toy_data):

    with pytest.raises(ValueError):
        ss = ShuffleSplit(n_splits=0)

    with pytest.raises(ValueError):
        ss = ShuffleSplit(test_size=10)
        next(ss.split(toy_data))

    with pytest.raises(ValueError):
        ss = ShuffleSplit(train_size=10)
        next(ss.split(toy_data))

    with pytest.raises(ValueError):
        ss = ShuffleSplit(test_size=3, train_size=3)
        next(ss.split(toy_data))

    with pytest.raises(ValueError):
        ss = ShuffleSplit(test_size=3, train_size=0)
        next(ss.split(toy_data))

    with pytest.raises(ValueError):
        ss = ShuffleSplit(test_size=0, train_size=3)
        next(ss.split(toy_data))

    # No need to cover the entire dataset
    ss = ShuffleSplit(test_size=1, train_size=1)
    next(ss.split(toy_data))

    # test test_size to int and train_size to None (complement)
    ss = ShuffleSplit(test_size=1)
    assert all(len(testset) == 1 for (_, testset) in ss.split(toy_data))
    assert all(trainset.n_ratings == 4 for (trainset, _) in ss.split(toy_data))

    # test test_size to float and train_size to None (complement)
    ss = ShuffleSplit(test_size=.2)  # 20% of 5 = 1
    assert all(len(testset) == 1 for (_, testset) in ss.split(toy_data))
    assert all(trainset.n_ratings == 4 for (trainset, _) in ss.split(toy_data))

    # test test_size to int and train_size to int
    ss = ShuffleSplit(test_size=2, train_size=2)
    assert all(len(testset) == 2 for (_, testset) in ss.split(toy_data))
    assert all(trainset.n_ratings == 2 for (trainset, _) in ss.split(toy_data))

    # test test_size to None (complement) and train_size to int
    ss = ShuffleSplit(test_size=None, train_size=2)
    assert all(len(testset) == 3 for (_, testset) in ss.split(toy_data))
    assert all(trainset.n_ratings == 2 for (trainset, _) in ss.split(toy_data))

    # test test_size to None (complement) and train_size to float
    ss = ShuffleSplit(test_size=None, train_size=.2)
    assert all(len(testset) == 4 for (_, testset) in ss.split(toy_data))
    assert all(trainset.n_ratings == 1 for (trainset, _) in ss.split(toy_data))

    # test default parameters: 5 splits, test_size = .2, train_size = None
    ss = ShuffleSplit()
    assert len(list(ss.split(toy_data))) == 5
    assert all(len(testset) == 1 for (_, testset) in ss.split(toy_data))
    assert all(trainset.n_ratings == 4 for (trainset, _) in ss.split(toy_data))

    # Test random_state parameter
    # If random_state is None, you get different split each time (conditioned
    # by rng of course)
    ss = ShuffleSplit(random_state=None)
    testsets_a = [testset for (_, testset) in ss.split(toy_data)]
    testsets_b = [testset for (_, testset) in ss.split(toy_data)]
    assert testsets_a != testsets_b
    # Repeated called to split when random_state is set lead to the same folds
    ss = ShuffleSplit(random_state=1)
    testsets_a = [testset for (_, testset) in ss.split(toy_data)]
    testsets_b = [testset for (_, testset) in ss.split(toy_data)]
    assert testsets_a == testsets_b

    # Test shuffle parameter, if False then splits are the same regardless of
    # random_state.
    ss = ShuffleSplit(random_state=1, shuffle=False)
    testsets_a = [testset for (_, testset) in ss.split(toy_data)]
    testsets_b = [testset for (_, testset) in ss.split(toy_data)]
    assert testsets_a == testsets_b


def test_train_test_split(toy_data):

    # test test_size to int and train_size to None (complement)
    trainset, testset = train_test_split(toy_data, test_size=2, train_size=None)
    assert len(testset) == 2
    assert trainset.n_ratings == 3

    # test test_size to float and train_size to None (complement)
    trainset, testset = train_test_split(toy_data, test_size=.2, train_size=None)
    assert len(testset) == 1
    assert trainset.n_ratings == 4

    # test test_size to int and train_size to int
    trainset, testset = train_test_split(toy_data, test_size=2, train_size=3)
    assert len(testset) == 2
    assert trainset.n_ratings == 3

    # test test_size to None (complement) and train_size to int
    trainset, testset = train_test_split(toy_data, test_size=None, train_size=2)
    assert len(testset) == 3
    assert trainset.n_ratings == 2

    # test test_size to None (complement) and train_size to float
    trainset, testset = train_test_split(toy_data, test_size=None, train_size=.2)
    assert len(testset) == 4
    assert trainset.n_ratings == 1

    # Test random_state parameter
    # If random_state is None, you get different split each time (conditioned
    # by rng of course)
    _, testset_a = train_test_split(toy_data, random_state=None)
    _, testset_b = train_test_split(toy_data, random_state=None)
    assert testset_a != testset_b

    # Repeated called to split when random_state is set lead to the same folds
    _, testset_a = train_test_split(toy_data, random_state=1)
    _, testset_b = train_test_split(toy_data, random_state=1)
    assert testset_a == testset_b

    # Test shuffle parameter, if False then splits are the same regardless of
    # random_state.
    _, testset_a = train_test_split(toy_data, random_state=1, shuffle=None)
    _, testset_b = train_test_split(toy_data, random_state=1, shuffle=None)
    assert testset_a == testset_b


def test_RepeatedCV(toy_data):

    # test n_splits and n_repeats parameters
    rkf = RepeatedKFold(n_splits=3, n_repeats=2)
    assert len(list(rkf.split(toy_data))) == 3 * 2
    rkf = RepeatedKFold(n_splits=3, n_repeats=4)
    assert len(list(rkf.split(toy_data))) == 3 * 4
    rkf = RepeatedKFold(n_splits=4, n_repeats=3)
    assert len(list(rkf.split(toy_data))) == 4 * 3

    # Make sure folds different between 2 repetitions (even if
    # random_state is set, random_state controls the whole sequence)
    rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=3)
    testsets = list(testset for (_, testset) in rkf.split(toy_data))
    for i in range(3):
        assert testsets[i] != testsets[i + 3]

    # Make sure folds are same when same cv iterator is called on same data (if
    # random_state is set)
    rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=3)
    testsets_a = list(testset for (_, testset) in rkf.split(toy_data))
    testsets_b = list(testset for (_, testset) in rkf.split(toy_data))
    assert testsets_a == testsets_b

    # Make sure folds are different when random_state is None
    rkf = RepeatedKFold(n_splits=3, n_repeats=2, random_state=None)
    testsets_a = list(testset for (_, testset) in rkf.split(toy_data))
    testsets_b = list(testset for (_, testset) in rkf.split(toy_data))
    assert testsets_a != testsets_b


def test_LeaveOneOut(toy_data):

    loo = LeaveOneOut()
    with pytest.raises(ValueError):
        next(loo.split(toy_data))  # each user only has 1 item so trainsets fail

    reader = Reader('ml-100k')

    data_path = (os.path.dirname(os.path.realpath(__file__)) +
                 '/u1_ml100k_test')
    data = Dataset.load_from_file(file_path=data_path, reader=reader)

    # Test random_state parameter
    # If random_state is None, you get different split each time (conditioned
    # by rng of course)
    loo = LeaveOneOut(random_state=None)
    testsets_a = [testset for (_, testset) in loo.split(data)]
    testsets_b = [testset for (_, testset) in loo.split(data)]
    assert testsets_a != testsets_b
    # Repeated called to split when random_state is set lead to the same folds
    loo = LeaveOneOut(random_state=1)
    testsets_a = [testset for (_, testset) in loo.split(data)]
    testsets_b = [testset for (_, testset) in loo.split(data)]
    assert testsets_a == testsets_b

    # Make sure only one rating per user is present in the testset
    loo = LeaveOneOut()
    for _, testset in loo.split(data):
        cnt = Counter([uid for (uid, _, _) in testset])
        assert all(val == 1 for val in cnt.values())

    # test the min_n_ratings parameter
    loo = LeaveOneOut(min_n_ratings=5)
    for trainset, _ in loo.split(data):
        assert all(len(ratings) >= 5 for ratings in trainset.ur.values())

    loo = LeaveOneOut(min_n_ratings=10)
    for trainset, _ in loo.split(data):
        assert all(len(ratings) >= 10 for ratings in trainset.ur.values())

    loo = LeaveOneOut(min_n_ratings=10000)  # too high
    with pytest.raises(ValueError):
        next(loo.split(data))


def test_PredifinedKFold():

    reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                    rating_scale=(1, 5))

    current_dir = os.path.dirname(os.path.realpath(__file__))
    folds_files = [(current_dir + '/custom_train',
                    current_dir + '/custom_test')]

    data = Dataset.load_from_folds(folds_files=folds_files, reader=reader)

    # Make sure rating files are read correctly
    pkf = PredefinedKFold()
    trainset, testset = next(pkf.split(data))
    assert trainset.n_ratings == 6
    assert len(testset) == 3


def test_get_cv():

    get_cv(None)
    get_cv(4)
    get_cv(KFold())

    with pytest.raises(ValueError):
        get_cv(23.2)
    with pytest.raises(ValueError):
        get_cv('bad')
