"""Module for testing the dump module."""


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

    with tempfile.NamedTemporaryFile() as tmp_file:
        dump.dump(tmp_file.name, algo=algo)
        dump.load(tmp_file.name)

    algo.fit(trainset)
    predictions = algo.test(testset)

    with tempfile.NamedTemporaryFile() as tmp_file:
        dump.dump(tmp_file.name, predictions, algo)
        predictions_dumped, algo_dumped = dump.load(tmp_file.name)

        assert predictions == predictions_dumped

        predictions_algo_dumped = algo_dumped.test(testset)
        if not isinstance(algo, NormalPredictor):  # predictions are random
            assert predictions == predictions_algo_dumped


def test_dump_nothing():
    """Ensure that by default None objects are dumped."""
    with tempfile.NamedTemporaryFile() as tmp_file:
        dump.dump(tmp_file.name)
        predictions, algo = dump.load(tmp_file.name)
        assert predictions is None
        assert algo is None
