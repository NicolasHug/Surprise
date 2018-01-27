'''The utils module contains the get_rng and _parse_version functions.'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numbers

import numpy as np
import scipy


def get_rng(random_state):
    '''Return a 'validated' RNG.

    If random_state is None, use RandomState singleton from numpy.  Else if
    it's an integer, consider it's a seed and initialized an rng with that
    seed. If it's already an rng, return it.
    '''
    if random_state is None:
        return np.random.mtrand._rand
    elif isinstance(random_state, (numbers.Integral, np.integer)):
        return np.random.RandomState(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state
    raise ValueError('Wrong random state. Expecting None, an int or a numpy '
                     'RandomState instance, got a '
                     '{}'.format(type(random_state)))

def _parse_version(version_string):
    version = []
    for x in version_string.split('.'):
        try:
            version.append(int(x))
        except ValueError:
            # x may be of the form dev-1ea1592
            version.append(x)
    return tuple(version)

sp_version = _parse_version(scipy.__version__)
