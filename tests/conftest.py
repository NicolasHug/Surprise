import pytest

from surprise import NormalPredictor
from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNBaseline
from surprise import SVD
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise import KNNWithZScore


@pytest.fixture()
def all_prediction_algorithms():
    return (NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans, KNNBaseline,
            SVD, SVDpp, NMF, SlopeOne, CoClustering, KNNWithZScore)
