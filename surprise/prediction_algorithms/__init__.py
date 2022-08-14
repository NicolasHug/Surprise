"""
The :mod:`prediction_algorithms` package includes the prediction algorithms
available for recommendation.

The available prediction algorithms are:

.. autosummary::
    :nosignatures:

    random_pred.NormalPredictor
    baseline_only.BaselineOnly
    knns.KNNBasic
    knns.KNNWithMeans
    knns.KNNWithZScore
    knns.KNNBaseline
    matrix_factorization.SVD
    matrix_factorization.SVDpp
    matrix_factorization.NMF
    slope_one.SlopeOne
    co_clustering.CoClustering
"""

from .algo_base import AlgoBase
from .baseline_only import BaselineOnly
from .co_clustering import CoClustering
from .knns import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from .matrix_factorization import NMF, SVD, SVDpp

from .predictions import Prediction, PredictionImpossible
from .random_pred import NormalPredictor
from .slope_one import SlopeOne

__all__ = [
    "AlgoBase",
    "NormalPredictor",
    "BaselineOnly",
    "KNNBasic",
    "KNNBaseline",
    "KNNWithMeans",
    "SVD",
    "SVDpp",
    "NMF",
    "SlopeOne",
    "CoClustering",
    "PredictionImpossible",
    "Prediction",
    "KNNWithZScore",
]
