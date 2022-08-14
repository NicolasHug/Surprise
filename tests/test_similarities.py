"""
Module for testing the similarity measures
"""



import random

import numpy as np

import surprise.similarities as sims

n_x = 8
yr_global = {
    0: [(0, 3), (1, 3), (2, 3), (5, 1),                 (6, 1.5), (7, 3)],  # noqa
    1: [(0, 4), (1, 4), (2, 4),                                         ],  # noqa
    2: [                (2, 5), (3, 2), (4, 3)                          ],  # noqa
    3: [(1, 1),         (2, 4), (3, 2), (4, 3), (5, 3), (6, 3.5), (7, 2)],  # noqa
    4: [(1, 5),         (2, 1),                 (5, 2), (6, 2.5), (7, 2.5)], # noqa
}


def test_cosine_sim():
    """Tests for the cosine similarity."""

    yr = yr_global.copy()

    # shuffle every rating list, to ensure the order in which ratings are
    # processed does not matter (it's important because it used to be error
    # prone when we were using itertools.combinations)
    for _, ratings in yr.items():
        random.shuffle(ratings)

    sim = sims.cosine(n_x, yr, min_support=1)

    # check symmetry and bounds (as ratings are > 0, cosine sim must be >= 0)
    for xi in range(n_x):
        assert sim[xi, xi] == 1
        for xj in range(n_x):
            assert sim[xi, xj] == sim[xj, xi]
            assert 0 <= sim[xi, xj] <= 1

    # on common items, users 0, 1 and 2 have the same ratings
    assert sim[0, 1] == 1
    assert sim[0, 2] == 1

    # for vectors with constant ratings (even if they're different constants),
    # cosine sim is necessarily 1
    assert sim[3, 4] == 1

    # pairs of users (0, 3)  have no common items
    assert sim[0, 3] == 0
    assert sim[0, 4] == 0

    # check for float point support and computation correctness
    dot_product56 = 1 * 1.5 + 3 * 3.5 + 2 * 2.5
    assert sim[5, 6] == (dot_product56 /
                         ((1 ** 2 + 3 ** 2 + 2 ** 2) *
                          (1.5 ** 2 + 3.5 ** 2 + 2.5 ** 2)) ** 0.5
                         )

    # ensure min_support is taken into account. Only users 1 and 2 have more
    # than 4 common ratings.
    sim = sims.cosine(n_x, yr, min_support=4)
    for i in range(n_x):
        for j in range(i + 1, n_x):
            if i != 1 and j != 2:
                assert sim[i, j] == 0


def test_msd_sim():
    """Tests for the MSD similarity."""

    yr = yr_global.copy()

    # shuffle every rating list, to ensure the order in which ratings are
    # processed does not matter (it's important because it used to be error
    # prone when we were using itertools.combinations)
    for _, ratings in yr.items():
        random.shuffle(ratings)

    sim = sims.msd(n_x, yr, min_support=1)

    # check symmetry and bounds. MSD sim must be in [0, 1]
    for xi in range(n_x):
        assert sim[xi, xi] == 1
        for xj in range(n_x):
            assert sim[xi, xj] == sim[xj, xi]
            assert 0 <= sim[xi, xj] <= 1

    # on common items, users 0, 1 and 2 have the same ratings
    assert sim[0, 1] == 1
    assert sim[0, 2] == 1

    # msd(3, 4) = mean(1^2, 1^2). sim = (1 / (1 + msd)) = 1 / 2
    assert sim[3, 4] == .5

    # pairs of users (0, 3)  have no common items
    assert sim[0, 3] == 0
    assert sim[0, 4] == 0

    # ensure min_support is taken into account. Only users 1 and 2 have more
    # than 4 common ratings.
    sim = sims.msd(n_x, yr, min_support=4)
    for i in range(n_x):
        for j in range(i + 1, n_x):
            if i != 1 and j != 2:
                assert sim[i, j] == 0


def test_pearson_sim():
    """Tests for the pearson similarity."""

    yr = yr_global.copy()

    # shuffle every rating list, to ensure the order in which ratings are
    # processed does not matter (it's important because it used to be error
    # prone when we were using itertools.combinations)
    for _, ratings in yr.items():
        random.shuffle(ratings)

    sim = sims.pearson(n_x, yr, min_support=1)
    # check symmetry and bounds. -1 <= pearson coeff <= 1
    for xi in range(n_x):
        assert sim[xi, xi] == 1
        for xj in range(n_x):
            assert sim[xi, xj] == sim[xj, xi]
            assert -1 <= sim[xi, xj] <= 1

    # on common items, users 0, 1 and 2 have the same ratings
    assert sim[0, 1] == 1
    assert sim[0, 2] == 1

    # for vectors with constant ratings, pearson sim is necessarily zero (as
    # ratings are centered)
    assert sim[3, 4] == 0
    assert sim[2, 3] == 0
    assert sim[2, 4] == 0

    # pairs of users (0, 3), have no common items
    assert sim[0, 3] == 0
    assert sim[0, 4] == 0

    # almost same ratings (just with an offset of 0.5)
    assert sim[5, 6] == 1

    # ratings vary in the same direction
    assert sim[2, 5] > 0

    # check for float point support and computation correctness
    mean6 = (1.5 + 3.5 + 2.5) / 3
    var6 = (1.5 - mean6) ** 2 + (3.5 - mean6) ** 2 + (2.5 - mean6) ** 2
    mean7 = (3 + 2 + 2.5) / 3
    var7 = (3 - mean7) ** 2 + (2 - mean7) ** 2 + (2.5 - mean7) ** 2
    num = sum([((1.5 - mean6) * (3 - mean7)),
               ((3.5 - mean6) * (2 - mean7)),
               ((2.5 - mean6) * (2.5 - mean7))
               ])
    assert sim[6, 7] == num / (var6 * var7) ** 0.5

    # ensure min_support is taken into account. Only users 1 and 2 have more
    # than 4 common ratings.
    sim = sims.pearson(n_x, yr, min_support=4)
    for i in range(n_x):
        for j in range(i + 1, n_x):
            if i != 1 and j != 2:
                assert sim[i, j] == 0


def test_pearson_baseline_sim():
    """Tests for the pearson_baseline similarity."""

    yr = yr_global.copy()

    # shuffle every rating list, to ensure the order in which ratings are
    # processed does not matter (it's important because it used to be error
    # prone when we were using itertools.combinations)
    for _, ratings in yr.items():
        random.shuffle(ratings)

    global_mean = 3  # fake
    x_biases = np.random.normal(0, 1, n_x)  # fake
    y_biases = np.random.normal(0, 1, 5)  # fake (there are 5 ys)
    sim = sims.pearson_baseline(n_x, yr, 1, global_mean, x_biases, y_biases)
    # check symmetry and bounds. -1 <= pearson coeff <= 1
    for xi in range(n_x):
        assert sim[xi, xi] == 1
        for xj in range(n_x):
            assert sim[xi, xj] == sim[xj, xi]
            assert -1 <= sim[xi, xj] <= 1

    # Note: as sim now depends on baselines, which depend on both users and
    # items ratings, it's now impossible to test assertions such that 'as users
    # have the same ratings, they should have a maximal similarity'. Both users
    # AND common items should have same ratings.

    # pairs of users (0, 3), have no common items
    assert sim[0, 3] == 0
    assert sim[0, 4] == 0

    # ensure min_support is taken into account. Only users 1 and 2 have more
    # than 4 common ratings.
    sim = sims.pearson_baseline(n_x, yr, 4, global_mean, x_biases, y_biases)
    for i in range(n_x):
        for j in range(i + 1, n_x):
            if i != 1 and j != 2:
                assert sim[i, j] == 0
