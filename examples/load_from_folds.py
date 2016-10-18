"""
This module descibes how to load a dataset when folds (for cross validation)
are predefined by files.
"""

from pyrec import BaselineOnly
from pyrec import Dataset
from pyrec import Reader
from pyrec import evaluate


files_dir = '/home/nico/.pyrec_data/ml-100k/ml-100k/' # change this

reader = Reader('ml-100k')

train_file = files_dir + 'u%d.base'
test_file = files_dir + 'u%d.test'
folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

data = Dataset.load_from_folds(folds_files, reader=reader)

algo = BaselineOnly()

evaluate(algo, data)
