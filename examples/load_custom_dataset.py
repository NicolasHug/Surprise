"""
This module descibes how to load a custom dataset from a single file.

As a custom dataset we will actually use the movielens-100k dataset, but act as
if it were not built-in.
"""

from pyrec import BaselineOnly
from pyrec import Dataset
from pyrec import Reader
from pyrec import evaluate


# path to dataset file
file_path = '/home/nico/.pyrec_data/ml-100k/ml-100k/u.data' # change this

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format :
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating timestamp', sep='\t')

data = Dataset.load_from_file(file_path, reader=reader)
data.split(n_folds=5)

# We'll use an algorithm that predicts baseline estimates.
algo = BaselineOnly()

# Evaluate performances of our algorithm on the dataset.
evaluate(algo, data)
