import sys
import os

# I'm not sure if this is good practice...
# TODO: check that
sys.path.insert(1, os.path.join(sys.path[0], '../'))

import pyximport
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()})
import similarities as sims

n_x = 7
yr = {
        0 : [(0, 3), (1, 3), (2, 3),                 (5, 1), (6, 2)],
        1 : [(0, 3), (1, 3), (2, 3),                               ],
        2 : [                (2, 5), (3, 2), (4, 2)                ],
        3 : [                        (3, 3), (4, 3), (5, 2), (6, 1)],
     }

def test_cosine_sim():
    sim = sims.cosine(n_x, yr)

    # check symetry and bounds (as ratings are > 0, cosine sim must be >= 0)
    for xi in range(n_x):
        assert sim[xi, xi] == 1
        for xj in range(n_x):
            assert sim[xi, xj] == sim[xj, xi]
            assert 0 <= sim[xi, xj] <= 1

    # on common items, users 0, 1, and 2 have the same ratings
    assert sim[0, 1] == 1
    assert sim[0, 2] == 1
    assert sim[1, 2] == 1

    # pairs of users (0, 3), (0, 4) and (1, 4) have no common items
    assert sim[0, 3] == 0
    assert sim[0, 4] == 0
    assert sim[1, 3] == 0
    assert sim[1, 4] == 0

    # pairs of users (2, 3) and (2, 4) have only 1 common item
    # for vectors of size is 1, cosine sim is necessarily 1
    assert sim[2, 3] == 1
    assert sim[2, 4] == 1

    # for vectors with constant ratings (even if they're different constants),
    # cosine sim is necessarily 1 as well
    assert sim[3, 4] == 1

    # non constant and different ratings: cosine sim must be in ]0, 1[
    assert 0 < sim[5, 6] < 1
