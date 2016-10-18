"""
This module descibes how to evaluate a dataset on predifined train and test
files.
"""

from pyrec import BaselineOnly
from pyrec import Dataset
from pyrec import Reader
from pyrec import evaluate


files_dir = '/home/nico/.pyrec_data/ml-100k/ml-100k/' # change this

# As we're loading a dataset from files, we need to define a reader.
# We're loading files u1.base and u1.test from the movielens 100k dataset, and
# each line has the following format :
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating timestamp', sep='\t')

# Actually, as the Movielens-100k dataset is builtin, it has a proper reader so
# the previous line is equivalent to
# reader = Reader('ml-100k')

folds_files = [(files_dir + '/u1.base', files_dir + '/u1.test')]
data = Dataset.load_from_folds(folds_files, reader=reader)

# We'll use an algorithm that predicts baseline estimates.
algo = BaselineOnly()

# Evaluate performances of our algorithm on the dataset.
evaluate(algo, data)
