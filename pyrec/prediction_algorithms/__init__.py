"""
The :mod:`prediction_algorithms` package includes the prediction algorithms
available for recommendation.
"""

from .random import Random
from .baseline_only import BaselineOnly
from .knns import KNNBasic
from .knns import KNNBaseline
from .knns import KNNWithMeans
from .analogy import Parall
from .analogy import Pattern
from .clone import CloneBruteforce
from .clone import CloneMeanDiff
from .clone import CloneKNNMeanDiff
