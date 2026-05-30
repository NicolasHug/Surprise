"""Module for testing the dump module."""


import os
import random
import tempfile

import pytest

from surprise import (
    BaselineOnly,
    CoClustering,
    dump,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    KNNWithZScore,
    NMF,
    NormalPredictor,
    SlopeOne,
    SVD,
    SVDpp,
)
from surprise.model_selection import PredefinedKFold


@pytest.mark.parametrize(
    "algo",
    (
        NormalPredictor(),
        BaselineOnly(),
        KNNBasic(),
        KNNWithMeans(),
        KNNBaseline(),
        SVD(),
        SVDpp(),
        NMF(),
        SlopeOne(),
        CoClustering(),
        KNNWithZScore(),
    ),
)
def test_dump(algo, u1_ml100k):
    """Train an algorithm, compute its predictions then dump them.
    Ensure that the predictions that are loaded back are the correct ones, and
    that the predictions of the dumped algorithm are also equal to the other
    ones."""

    random.seed(0)

    trainset, testset = next(PredefinedKFold().split(u1_ml100k))

    fd, tmp_path = tempfile.mkstemp()
    os.close(fd)
    try:
        dump.dump(tmp_path, algo=algo)
        dump.load(tmp_path)
    finally:
        os.remove(tmp_path)

    algo.fit(trainset)
    predictions = algo.test(testset)

    fd, tmp_path = tempfile.mkstemp()
    os.close(fd)
    try:
        dump.dump(tmp_path, predictions, algo)
        predictions_dumped, algo_dumped = dump.load(tmp_path)

        assert predictions == predictions_dumped

        predictions_algo_dumped = algo_dumped.test(testset)
        if not isinstance(algo, NormalPredictor):  # predictions are random
            assert predictions == predictions_algo_dumped
    finally:
        os.remove(tmp_path)


def test_dump_nothing():
    """Ensure that by default None objects are dumped."""
    fd, tmp_path = tempfile.mkstemp()
    os.close(fd)
    try:
        dump.dump(tmp_path)
        predictions, algo = dump.load(tmp_path)
        assert predictions is None
        assert algo is None
    finally:
        os.remove(tmp_path)
