"""Module for testing the dump module."""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pytest

from surprise.prediction_algorithms.predictions import Prediction
from surprise import dump


def test_dump():

    p = Prediction(None, None, None, None, None)  # dummy prediction
    with pytest.raises(Exception):
        dump('wrong/file', [p])
