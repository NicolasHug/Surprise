"""
This module descibes how to train on a full dataset (when no testset is
built/specified) and how to query for specific predictions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import KNNBasic
from surprise import Dataset
from surprise import evaluate

# Load the movielens-100k dataset and split it into 3 folds for
# cross-validation.
data = Dataset.load_builtin('ml-100k')

# Retrieve the trainset.
trainset = data.build_full_trainset()

# Build an algorithm, and train it.
algo = KNNBasic()
algo.train(trainset)


##########################################
# we can now query for specific predicions

uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
iid = str(302)  # raw item id (as in the ratings file). They are **strings**!

# get a prediction for specific users and items.
pred = algo.predict(uid, iid, r_ui=4, verbose=True)


##########################################
# Tired? You can still call the 'split' method!
data.split(n_folds=3)
evaluate(algo, data)
