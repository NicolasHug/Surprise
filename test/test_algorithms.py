from pyrec.prediction_algorithms import *
from pyrec.dataset import Dataset
from pyrec.dataset import Reader
from pyrec.evaluate import evaluate

algorithms = (NormalPredictor, BaselineOnly, KNNBasic, KNNWithMeans,
              KNNBaseline)

train_file = 'datasets/ml-100k/ml-100k/u1.base'
test_file = 'datasets/ml-100k/ml-100k/u1.test'
data = Dataset.load_from_files(train_file, test_file, Reader('ml-100k'))

def test_algos():
    for klass in algorithms:
        algo = klass()
        evaluate(algo, data)


