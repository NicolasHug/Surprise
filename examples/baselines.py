"""
This module gives an example of how to configure baseline estimates
computation.
"""

from pyrec import BaselineOnly
from pyrec import Dataset
from pyrec import evaluate


# Load the movielens-100k dataset.
data = Dataset.load_builtin('ml-100k')

# Example using ALS
print('Using ALS')
bsl_options = {'method' : 'als',
               'n_epochs' : 5,
               'reg_u' : 12,
               'reg_i' : 5
}
algo = BaselineOnly(bsl_options=bsl_options)

evaluate(algo, data)

# Example using SGD
print('Using SGD')
bsl_options = {'method' : 'sgd',
               'learning_rate' : .00005,
}
algo = BaselineOnly(bsl_options=bsl_options)

evaluate(algo, data)


# TODO: make example with pearson baseline sim

