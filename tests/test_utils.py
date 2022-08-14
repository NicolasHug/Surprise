


import numpy as np
import pytest

from surprise.utils import get_rng


def test_get_rng():

    # assert two RNG with same int are the same
    rng_a = get_rng(12)
    rng_b = get_rng(12)
    a = [rng_a.rand() for _ in range(10)]
    b = [rng_b.rand() for _ in range(10)]
    assert a == b

    # assert passing an int returns the corresponding numpy rng instance
    rng_a = get_rng(12)
    rng_b = np.random.RandomState(12)

    a = [rng_a.rand() for _ in range(10)]
    b = [rng_b.rand() for _ in range(10)]
    assert a == b

    # Make sure this is ok
    get_rng(None)

    with pytest.raises(ValueError):
        get_rng(23.2)
    with pytest.raises(ValueError):
        get_rng('bad')
