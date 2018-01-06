from pkg_resources import get_distribution

from .prediction_algorithms import AlgoBase
from .prediction_algorithms import NormalPredictor
from .prediction_algorithms import BaselineOnly
from .prediction_algorithms import KNNBasic
from .prediction_algorithms import KNNWithMeans
from .prediction_algorithms import KNNWithZScore
from .prediction_algorithms import KNNBaseline
from .prediction_algorithms import SVD
from .prediction_algorithms import SVDpp
from .prediction_algorithms import NMF
from .prediction_algorithms import SlopeOne
from .prediction_algorithms import CoClustering

from .prediction_algorithms import PredictionImpossible
from .prediction_algorithms import Prediction

from .dataset import Dataset
from .reader import Reader
from .trainset import Trainset
from .builtin_datasets import get_dataset_dir
from .evaluate import evaluate
from .evaluate import print_perf
from .evaluate import GridSearch
from . import model_selection
from . import dump

__all__ = ['AlgoBase', 'NormalPredictor', 'BaselineOnly', 'KNNBasic',
           'KNNWithMeans', 'KNNBaseline', 'SVD', 'SVDpp', 'NMF', 'SlopeOne',
           'CoClustering', 'PredictionImpossible', 'Prediction', 'Dataset',
           'Reader', 'Trainset', 'evaluate', 'print_perf', 'GridSearch',
           'dump', 'KNNWithZScore', 'get_dataset_dir', 'model_selection']

__version__ = get_distribution('scikit-surprise').version
