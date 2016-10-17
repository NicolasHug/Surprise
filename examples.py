import random

from pyrec.dataset import Dataset
from pyrec.dataset import Reader
from pyrec.prediction_algorithms import NormalPredictor
from pyrec.prediction_algorithms import BaselineOnly
from pyrec.prediction_algorithms import KNNBasic
from pyrec.prediction_algorithms import KNNWithMeans
from pyrec.prediction_algorithms import KNNBaseline
from pyrec.evaluate import evaluate


random.seed(0)

#algo = NormalPredictor(user_based=True)
#algo = BaselineOnly()
algo = KNNBasic(sim=dict(name='cos'))
#algo = KNNWithMeans(user_based=True)
#algo = KNNBaseline(sim=dict(name='pearson_baseline', shrinkage=100))

#reader = Reader(line_format='user item rating timestamp', sep='\t')
#reader = Reader('BX')

"""
train_file = '/home/nico/dev/pyrec/pyrec/datasets/ml-100k/ml-100k/u%d.base'
test_file = '/home/nico/dev/pyrec/pyrec/datasets/ml-100k/ml-100k/u%d.test'
folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]
data = Dataset.load_from_folds(folds_files, reader=reader)
"""

data = Dataset.load('ml-100k')
data.split(n_folds=5)
"""

data = Dataset.load_from_files(
       '/home/nico/.pyrec_data/ml-100k/ml-100k/u1.base',
       '/home/nico/.pyrec_data/ml-100k/ml-100k/u1.test',
       reader=Reader('ml-100k'))
"""

evaluate(algo, data, with_dump=True)

"""
# this is like... cool
algo = KNNBaseline()
data = Dataset.load('ml-100k')
data.split(n_folds=5)
evaluate(algo, data)
"""
