"""Module for testing the dump module."""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import tempfile

from surprise.prediction_algorithms.predictions import Prediction
from surprise.prediction_algorithms.algo_base import AlgoBase
from surprise.dataset import Trainset
from surprise import dump


def test_dump():

    predictions = [Prediction(None, None, None, None, None)]
    algo = AlgoBase()
    trainset = Trainset(*[None] * 9)

    with tempfile.NamedTemporaryFile() as tmp_file:
        dump(tmp_file.name, predictions, trainset, algo)
