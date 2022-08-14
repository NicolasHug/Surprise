"""The utils module contains the get_rng function."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numbers

import numpy as np


def get_rng(random_state):
    """Return a 'validated' RNG.

    If random_state is None, use RandomState singleton from numpy.  Else if
    it's an integer, consider it's a seed and initialized an rng with that
    seed. If it's already an rng, return it.
    """
    if random_state is None:
        return np.random.mtrand._rand
    elif isinstance(random_state, (numbers.Integral, np.integer)):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError('Wrong random state. Expecting None, an int or a numpy '
                     'RandomState instance, got a '
                     '{}'.format(type(random_state)))
