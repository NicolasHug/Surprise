"""
This module gives an example of how to configure similarity measures
computation.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import KNNBasic
from surprise import Dataset
from surprise.model_selection import cross_validate


# Load the movielens-100k dataset.
data = Dataset.load_builtin('ml-100k')

# Example using cosine similarity
sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBasic(sim_options=sim_options)

cross_validate(algo, data, verbose=True)

# Example using pearson_baseline similarity
sim_options = {'name': 'pearson_baseline',
               'shrinkage': 0  # no shrinkage
               }
algo = KNNBasic(sim_options=sim_options)

cross_validate(algo, data, verbose=True)
