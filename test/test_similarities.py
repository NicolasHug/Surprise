"""
Module for testing the similarity measures
"""

import sys
import os
import random
import copy

import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()})

import pyrec.similarities as sims
from pyrec.prediction_algorithms import BaselineOnly

n_x = 7
yr_global = {
        0 : [(0, 3), (1, 3), (2, 3),                 (5, 1), (6, 2)],
        1 : [(0, 4), (1, 4), (2, 4),                               ],
        2 : [                (2, 5), (3, 2), (4, 3)                ],
        3 : [        (1, 1), (2, 4), (3, 2), (4, 3), (5, 3), (6, 4)],
        4 : [        (1, 5), (2, 1),                 (5, 2), (6, 3)],
            }

def test_cosine_sim():

    yr = yr_global.copy()

    # shuffle every rating list, to ensure the order in which ratings are
    # processed does not matter (it's important because it used to be error
    # prone when we were using itertools.combinations)
    for _, ratings in yr.items():
        random.shuffle(ratings)

    sim = sims.cosine(n_x, yr)

    # check symetry and bounds (as ratings are > 0, cosine sim must be >= 0)
    for xi in range(n_x):
        assert sim[xi, xi] == 1
        for xj in range(n_x):
            assert sim[xi, xj] == sim[xj, xi]
            assert 0 <= sim[xi, xj] <= 1

    # on common items, users 0 and 1 have the same ratings
    assert sim[0, 1] == 1
    assert sim[0, 2] == 1

    # for vectors with constant ratings (even if they're different constants),
    # cosine sim is necessarily 1
    assert sim[3, 4] == 1

    # pairs of users (0, 3)  have no common items
    assert sim[0, 3] == 0
    assert sim[0, 4] == 0


    # non constant and different ratings: cosine sim must be in ]0, 1[
    assert 0 < sim[5, 6] < 1


def test_pearson_sim():

    yr = yr_global.copy()

    # shuffle every rating list, to ensure the order in which ratings are
    # processed does not matter (it's important because it used to be error
    # prone when we were using itertools.combinations)
    for _, ratings in yr.items():
        random.shuffle(ratings)

    sim = sims.pearson(n_x, yr)
    # check symetry and bounds. -1 <= pearson coeff <= 1
    for xi in range(n_x):
        assert sim[xi, xi] == 1
        for xj in range(n_x):
            assert sim[xi, xj] == sim[xj, xi]
            assert -1 <= sim[xi, xj] <= 1

    # on common items, users 0, and 1 have the same ratings
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

    # almost same ratings (just with an offset of 1)
    assert sim[5, 6] == 1

    # ratings vary in the same direction
    assert sim[2, 5] > 0
    # ratings vary in inverse direction
    assert sim[1, 5] < 0
