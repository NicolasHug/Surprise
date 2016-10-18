"""
This module descibes the most basic usage of Pyrec: you define a prediction
algorithm, (down)load a dataset and evaluate the performances of the algorithm.
"""

from pyrec import NormalPredictor
from pyrec import Dataset
from pyrec import evaluate


# Load the movielens-100k dataset and split it into 5 folds for
# cross-validation.
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=5)

# We'll use an algorithm that predicts a random rating.
algo = NormalPredictor()

# Evaluate performances of our algorithm on the dataset.
evaluate(algo, data)
