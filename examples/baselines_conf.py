"""
This module gives an example of how to configure baseline estimates
computation.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import BaselineOnly
from surprise import KNNBasic
from surprise import Dataset
from surprise import evaluate


# Load the movielens-100k dataset.
data = Dataset.load_builtin('ml-100k')

# Example using ALS
print('Using ALS')
bsl_options = {'method': 'als',
               'n_epochs': 5,
               'reg_u': 12,
               'reg_i': 5
               }
algo = BaselineOnly(bsl_options=bsl_options)

evaluate(algo, data)

# Example using SGD
print('Using SGD')
bsl_options = {'method': 'sgd',
               'learning_rate': .00005,
               }
algo = BaselineOnly(bsl_options=bsl_options)

evaluate(algo, data)

# Some similarity measures may use baselines. It works just the same.
print('Using ALS with pearson_baseline similarity')
bsl_options = {'method': 'als',
               'n_epochs': 20,
               }
sim_options = {'name': 'pearson_baseline'}
algo = KNNBasic(bsl_options=bsl_options, sim_options=sim_options)

evaluate(algo, data)
