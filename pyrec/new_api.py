import stats
from dataset import Dataset
from dataset import Reader
from prediction_algorithms import NormalPredictor
from prediction_algorithms import BaselineOnly
from prediction_algorithms import KNNBasic
from prediction_algorithms import KNNWithMeans
from prediction_algorithms import KNNBaseline

def evaluate(algo, data):
    for trainset, testset in data.folds:
        algo.train(trainset)
        algo.test(testset)
        stats.compute_stats(algo.preds)


#algo = NormalPredictor(user_based=True)
#algo = BaselineOnly(user_based=True)
#algo = KNNBasic(user_based=True)
#algo = KNNWithMeans(user_based=True)
algo = KNNBaseline(user_based=True, sim_name='pearson_baseline', shrinkage=100)

reader = Reader(line_format='user item rating timestamp', sep='\t')
"""
train_name = '/home/nico/dev/pyrec/pyrec/datasets/ml-100k/ml-100k/u1.base'
test_name = '/home/nico/dev/pyrec/pyrec/datasets/ml-100k/ml-100k/u1.test'
data = Dataset.load_from_files(train_name, test_name, reader=reader)
"""
file_name = '/home/nico/dev/pyrec/pyrec/datasets/ml-100k/ml-100k/u.data'
data = Dataset.load_from_file(file_name, reader)
data.split(n_folds=5)

evaluate(algo, data)
