"""Module for testing the dump module."""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import tempfile
import random
import os

from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise import dump


def test_dump():
    """Train an algorithm, compute its predictions then dump them.
    Ensure that the predictions that are loaded back are the correct ones, and
    that the predictions of the dumped algorithm are also equal to the other
    ones."""

    random.seed(0)

    train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
    test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
    data = Dataset.load_from_folds([(train_file, test_file)],
                                   Reader('ml-100k'))

    for trainset, testset in data.folds():
        pass

    algo = BaselineOnly()
    algo.train(trainset)
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
