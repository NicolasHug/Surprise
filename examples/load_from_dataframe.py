"""
This module descibes how to load a dataset from a pandas dataframe.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import pandas as pd

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader


# Dummy algo
algo = NormalPredictor()

# Creation of the dataframe. Column names are irrelevant.
ratings_dict = {'itemID': [1, 1, 1, 2, 2],
                'userID': [9, 32, 2, 45, 'user_foo'],
                'rating': [3, 2, 4, 3, 1]}
df = pd.DataFrame(ratings_dict)

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
data.split(2)  # data can now be used normally

for trainset, testset in data.folds():
    algo.train(trainset)
    algo.test(testset)
