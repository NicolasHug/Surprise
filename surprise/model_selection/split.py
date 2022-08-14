"""
The :mod:`model_selection.split<surprise.model_selection.split>` module
contains various cross-validation iterators. Design and tools are inspired from
the mighty scikit learn.

The available iterators are:

.. autosummary::
    :nosignatures:

    KFold
    RepeatedKFold
    ShuffleSplit
    LeaveOneOut
    PredefinedKFold

This module also contains a function for splitting datasets into trainset and
testset:

.. autosummary::
    :nosignatures:

    train_test_split

"""

import numbers
from collections import defaultdict
from itertools import chain
from math import ceil, floor

import numpy as np

from ..utils import get_rng


def get_cv(cv):
    """Return a 'validated' CV iterator."""

    if cv is None:
        return KFold(n_splits=5)
    if isinstance(cv, numbers.Integral):
        return KFold(n_splits=cv)
    if hasattr(cv, "split") and not isinstance(cv, str):
        return cv  # str have split

    raise ValueError(
        "Wrong CV object. Expecting None, an int or CV iterator, "
        "got a {}".format(type(cv))
    )


class KFold:
    """A basic cross-validation iterator.

    Each fold is used once as a testset while the k - 1 remaining folds are
    used for training.

    See an example in the :ref:`User Guide <use_cross_validation_iterators>`.

    Args:
        n_splits(int): The number of folds.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for determining the folds. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same splits over multiple calls to ``split()``.
            If RandomState instance, this same instance is used as RNG. If
            ``None``, the current RNG from numpy is used. ``random_state`` is
            only used if ``shuffle`` is ``True``.  Default is ``None``.
        shuffle(bool): Whether to shuffle the ratings in the ``data`` parameter
            of the ``split()`` method. Shuffling is not done in-place. Default
            is ``True``.
    """

    def __init__(self, n_splits=5, random_state=None, shuffle=True):

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, data):
        """Generator function to iterate over trainsets and testsets.

        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be divided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        """

        if self.n_splits > len(data.raw_ratings) or self.n_splits < 2:
            raise ValueError(
                "Incorrect value for n_splits={}. "
                "Must be >=2 and less than the number "
                "of ratings".format(len(data.raw_ratings))
            )

        # We use indices to avoid shuffling the original data.raw_ratings list.
        indices = np.arange(len(data.raw_ratings))

        if self.shuffle:
            get_rng(self.random_state).shuffle(indices)

        start, stop = 0, 0
        for fold_i in range(self.n_splits):
            start = stop
            stop += len(indices) // self.n_splits
            if fold_i < len(indices) % self.n_splits:
                stop += 1

            raw_trainset = [
                data.raw_ratings[i] for i in chain(indices[:start], indices[stop:])
            ]
            raw_testset = [data.raw_ratings[i] for i in indices[start:stop]]

            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

    def get_n_folds(self):

        return self.n_splits


class RepeatedKFold:
    """
    Repeated :class:`KFold` cross validator.

    Repeats :class:`KFold` n times with different randomization in each
    repetition.

    See an example in the :ref:`User Guide <use_cross_validation_iterators>`.

    Args:
        n_splits(int): The number of folds.
        n_repeats(int): The number of repetitions.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for determining the folds. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same splits over multiple calls to ``split()``.
            If RandomState instance, this same instance is used as RNG. If
            ``None``, the current RNG from numpy is used. ``random_state`` is
            only used if ``shuffle`` is ``True``.  Default is ``None``.
        shuffle(bool): Whether to shuffle the ratings in the ``data`` parameter
            of the ``split()`` method. Shuffling is not done in-place. Default
            is ``True``.
    """

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):

        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_splits = n_splits

    def split(self, data):
        """Generator function to iterate over trainsets and testsets.

        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be divided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        """

        rng = get_rng(self.random_state)

        for _ in range(self.n_repeats):
            cv = KFold(n_splits=self.n_splits, random_state=rng, shuffle=True)
            yield from cv.split(data)

    def get_n_folds(self):

        return self.n_repeats * self.n_splits


class ShuffleSplit:
    """A basic cross-validation iterator with random trainsets and testsets.

    Contrary to other cross-validation strategies, random splits do not
    guarantee that all folds will be different, although this is still very
    likely for sizeable datasets.

    See an example in the :ref:`User Guide <use_cross_validation_iterators>`.

    Args:
        n_splits(int): The number of folds.
        test_size(float or int ``None``): If float, it represents the
            proportion of ratings to include in the testset. If int,
            represents the absolute number of ratings in the testset. If
            ``None``, the value is set to the complement of the trainset size.
            Default is ``.2``.
        train_size(float or int or ``None``): If float, it represents the
            proportion of ratings to include in the trainset. If int,
            represents the absolute number of ratings in the trainset. If
            ``None``, the value is set to the complement of the testset size.
            Default is ``None``.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for determining the folds. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same splits over multiple calls to ``split()``.
            If RandomState instance, this same instance is used as RNG. If
            ``None``, the current RNG from numpy is used. ``random_state`` is
            only used if ``shuffle`` is ``True``.  Default is ``None``.
        shuffle(bool): Whether to shuffle the ratings in the ``data`` parameter
            of the ``split()`` method. Shuffling is not done in-place. Setting
            this to `False` defeats the purpose of this iterator, but it's
            useful for the implementation of :func:`train_test_split`. Default
            is ``True``.
    """

    def __init__(
        self,
        n_splits=5,
        test_size=0.2,
        train_size=None,
        random_state=None,
        shuffle=True,
    ):

        if n_splits <= 0:
            raise ValueError(
                "n_splits = {} should be strictly greater than " "0.".format(n_splits)
            )
        if test_size is not None and test_size <= 0:
            raise ValueError(
                "test_size={} should be strictly greater than " "0".format(test_size)
            )

        if train_size is not None and train_size <= 0:
            raise ValueError(
                "train_size={} should be strictly greater than " "0".format(train_size)
            )

        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.random_state = random_state
        self.shuffle = shuffle

    def validate_train_test_sizes(self, test_size, train_size, n_ratings):

        if test_size is not None and test_size >= n_ratings:
            raise ValueError(
                "test_size={} should be less than the number of "
                "ratings {}".format(test_size, n_ratings)
            )

        if train_size is not None and train_size >= n_ratings:
            raise ValueError(
                "train_size={} should be less than the number of"
                " ratings {}".format(train_size, n_ratings)
            )

        if np.asarray(test_size).dtype.kind == "f":
            test_size = ceil(test_size * n_ratings)

        if train_size is None:
            train_size = n_ratings - test_size
        elif np.asarray(train_size).dtype.kind == "f":
            train_size = floor(train_size * n_ratings)

        if test_size is None:
            test_size = n_ratings - train_size

        if train_size + test_size > n_ratings:
            raise ValueError(
                "The sum of train_size and test_size ({}) "
                "should be smaller than the number of "
                "ratings {}.".format(train_size + test_size, n_ratings)
            )

        return int(train_size), int(test_size)

    def split(self, data):
        """Generator function to iterate over trainsets and testsets.

        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be divided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        """

        test_size, train_size = self.validate_train_test_sizes(
            self.test_size, self.train_size, len(data.raw_ratings)
        )
        rng = get_rng(self.random_state)

        for _ in range(self.n_splits):

            if self.shuffle:
                permutation = rng.permutation(len(data.raw_ratings))
            else:
                permutation = np.arange(len(data.raw_ratings))

            raw_trainset = [data.raw_ratings[i] for i in permutation[:test_size]]
            raw_testset = [
                data.raw_ratings[i]
                for i in permutation[test_size : (test_size + train_size)]
            ]

            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

    def get_n_folds(self):

        return self.n_splits


def train_test_split(
    data, test_size=0.2, train_size=None, random_state=None, shuffle=True
):
    """Split a dataset into trainset and testset.

    See an example in the :ref:`User Guide <train_test_split_example>`.

    Note: this function cannot be used as a cross-validation iterator.

    Args:
        data(:obj:`Dataset <surprise.dataset.Dataset>`): The dataset to split
            into trainset and testset.
        test_size(float or int ``None``): If float, it represents the
            proportion of ratings to include in the testset. If int,
            represents the absolute number of ratings in the testset. If
            ``None``, the value is set to the complement of the trainset size.
            Default is ``.2``.
        train_size(float or int or ``None``): If float, it represents the
            proportion of ratings to include in the trainset. If int,
            represents the absolute number of ratings in the trainset. If
            ``None``, the value is set to the complement of the testset size.
            Default is ``None``.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for determining the folds. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same splits over multiple calls to ``split()``.
            If RandomState instance, this same instance is used as RNG. If
            ``None``, the current RNG from numpy is used. ``random_state`` is
            only used if ``shuffle`` is ``True``.  Default is ``None``.
        shuffle(bool): Whether to shuffle the ratings in the ``data``
            parameter. Shuffling is not done in-place. Default is ``True``.
    """
    ss = ShuffleSplit(
        n_splits=1,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    return next(ss.split(data))


class LeaveOneOut:
    """Cross-validation iterator where each user has exactly one rating in the
    testset.

    Contrary to other cross-validation strategies, ``LeaveOneOut`` does not
    guarantee that all folds will be different, although this is still very
    likely for sizeable datasets.

    See an example in the :ref:`User Guide <use_cross_validation_iterators>`.

    Args:
        n_splits(int): The number of folds.
        random_state(int, RandomState instance from numpy, or ``None``):
            Determines the RNG that will be used for determining the folds. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same splits over multiple calls to ``split()``.
            If RandomState instance, this same instance is used as RNG. If
            ``None``, the current RNG from numpy is used. ``random_state`` is
            only used if ``shuffle`` is ``True``.  Default is ``None``.
        min_n_ratings(int): Minimum number of ratings for each user in the
            trainset. E.g. if ``min_n_ratings`` is ``2``, we are sure each user
            has at least ``2`` ratings in the trainset (and ``1`` in the
            testset). Other users are discarded. Default is ``0``, so some
            users (having only one rating) may be in the testset and not in the
            trainset.
    """

    def __init__(self, n_splits=5, random_state=None, min_n_ratings=0):

        self.n_splits = n_splits
        self.random_state = random_state
        self.min_n_ratings = min_n_ratings

    def split(self, data):
        """Generator function to iterate over trainsets and testsets.

        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be divided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        """

        # map ratings to the users ids
        user_ratings = defaultdict(list)
        for uid, iid, r_ui, _ in data.raw_ratings:
            user_ratings[uid].append((uid, iid, r_ui, None))

        rng = get_rng(self.random_state)

        for _ in range(self.n_splits):
            # for each user, randomly choose a rating and put it in the
            # testset.
            raw_trainset, raw_testset = [], []
            for uid, ratings in user_ratings.items():
                if len(ratings) > self.min_n_ratings:
                    i = rng.randint(0, len(ratings))
                    raw_testset.append(ratings[i])
                    raw_trainset += [
                        rating for (j, rating) in enumerate(ratings) if j != i
                    ]

            if not raw_trainset:
                raise ValueError(
                    "Could not build any trainset. Maybe " "min_n_ratings is too high?"
                )
            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

    def get_n_folds(self):

        return self.n_splits


class PredefinedKFold:
    """A cross-validation iterator to when a dataset has been loaded with the
    :meth:`load_from_folds <surprise.dataset.Dataset.load_from_folds>`
    method.

    See an example in the :ref:`User Guide <load_from_folds_example>`.
    """

    def split(self, data):
        """Generator function to iterate over trainsets and testsets.

        Args:
            data(:obj:`Dataset<surprise.dataset.Dataset>`): The data containing
                ratings that will be divided into trainsets and testsets.

        Yields:
            tuple of (trainset, testset)
        """

        self.n_splits = len(data.folds_files)
        for train_file, test_file in data.folds_files:

            raw_trainset = data.read_ratings(train_file)
            raw_testset = data.read_ratings(test_file)
            trainset = data.construct_trainset(raw_trainset)
            testset = data.construct_testset(raw_testset)

            yield trainset, testset

    def get_n_folds(self):

        return self.n_splits
