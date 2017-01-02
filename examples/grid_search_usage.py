"""
This module describes how to manually train and test an algorithm without using
the evaluate() function.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise.evaluate import GridSearch
from surprise.prediction_algorithms import SVD
from surprise.dataset import Dataset

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}

gridSearch = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])

# Prepare Data
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)

gridSearch.evaluate(data)

# best RMSE score
print(gridSearch.best_score['RMSE'])
# combination of parameters that gave the best RMSE score
print(gridSearch.best_params['RMSE'])

# best FCP score
print(gridSearch.best_score['FCP'])
# combination of parameters that gave the best FCP score
print(gridSearch.best_params['FCP'])

import pandas as pd

results_df = pd.DataFrame.from_dict(gridSearch.cv_results)
print(results_df)
