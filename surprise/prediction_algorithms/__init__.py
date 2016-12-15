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
"""

from .algo_base import AlgoBase
from .random_pred import NormalPredictor
from .baseline_only import BaselineOnly
from .knns import KNNBasic
from .knns import KNNBaseline
from .knns import KNNWithMeans
from .matrix_factorization import SVD
from .matrix_factorization import SVDpp
from .slope_one import SlopeOne

from .predictions import PredictionImpossible

__all__ = ['AlgoBase', 'NormalPredictor', 'BaselineOnly', 'KNNBasic',
           'KNNBaseline', 'KNNWithMeans', 'SVD', 'SVDpp', 'SlopeOne',
           'PredictionImpossible']
