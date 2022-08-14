"""Module for testing the dump module."""



import tempfile
import random

from surprise import BaselineOnly
from surprise import dump
from surprise.model_selection import PredefinedKFold


def test_dump(u1_ml100k):
    """Train an algorithm, compute its predictions then dump them.
    Ensure that the predictions that are loaded back are the correct ones, and
    that the predictions of the dumped algorithm are also equal to the other
    ones."""

    random.seed(0)

    trainset, testset = next(PredefinedKFold().split(u1_ml100k))

    algo = BaselineOnly()
    algo.fit(trainset)
    predictions = algo.test(testset)

    with tempfile.NamedTemporaryFile() as tmp_file:
        dump.dump(tmp_file.name, predictions, algo)
        predictions_dumped, algo_dumped = dump.load(tmp_file.name)

        predictions_algo_dumped = algo_dumped.test(testset)
        assert predictions == predictions_dumped
        assert predictions == predictions_algo_dumped


def test_dump_nothing():
    """Ensure that by default None objects are dumped."""
    with tempfile.NamedTemporaryFile() as tmp_file:
        dump.dump(tmp_file.name)
        predictions, algo = dump.load(tmp_file.name)
        assert predictions is None
        assert algo is None
