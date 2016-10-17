import os

from pyrec.prediction_algorithms import *
from pyrec.dataset import Dataset
from pyrec.dataset import Reader
from pyrec.evaluate import evaluate


algorithms = (NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans,
              KNNBaseline)

# the test and train files are from the ml-100k dataset (10% of u1.base and
# 10 % of u1.test)
train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
data = Dataset.load_from_files(train_file, test_file, Reader('ml-100k'))

def test_algos():
    for klass in algorithms:
        algo = klass()
        evaluate(algo, data)
