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
    knns.KNNBaseline
    matrix_factorization.SVD
    matrix_factorization.SVDpp
    matrix_factorization.NMF
    slope_one.SlopeOne
    co_clustering.CoClustering
"""

from .algo_base import AlgoBase
from .random_pred import NormalPredictor
from .baseline_only import BaselineOnly
from .knns import KNNBasic
from .knns import KNNBaseline
from .knns import KNNWithMeans
from .matrix_factorization import SVD
from .matrix_factorization import SVDpp
from .matrix_factorization import NMF
from .slope_one import SlopeOne
from .co_clustering import CoClustering

from .predictions import PredictionImpossible

__all__ = ['AlgoBase', 'NormalPredictor', 'BaselineOnly', 'KNNBasic',
           'KNNBaseline', 'KNNWithMeans', 'SVD', 'SVDpp', 'NMF', 'SlopeOne',
           'CoClustering', 'PredictionImpossible']
