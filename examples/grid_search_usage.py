"""
This module describes how to manually train and test an algorithm without using
the evaluate() function.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import GridSearch
from surprise import SVD
from surprise import Dataset

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}

grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])

# Prepare Data
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)

grid_search.evaluate(data)

# best RMSE score
print(grid_search.best_score['RMSE'])
# >>> 0.96117566386

# combination of parameters that gave the best RMSE score
print(grid_search.best_params['RMSE'])
# >>> {'reg_all': 0.4, 'lr_all': 0.005, 'n_epochs': 10}

# best FCP score
print(grid_search.best_score['FCP'])
# >>> 0.702279736531

# combination of parameters that gave the best FCP score
print(grid_search.best_params['FCP'])
# >>> {'reg_all': 0.6, 'lr_all': 0.005, 'n_epochs': 10}

import pandas as pd  # noqa

results_df = pd.DataFrame.from_dict(grid_search.cv_results)
print(results_df)
