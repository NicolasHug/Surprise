"""
This module is used for generating the doc tables about the
GridSearchCV.cv_results attribute.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from tabulate import tabulate

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import GridSearchCV

# Use movielens-100K
data = Dataset.load_builtin('ml-100k')

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

table = [[] for _ in range(len(gs.cv_results['params']))]
for i in range(len(gs.cv_results['params'])):
    for key in gs.cv_results.keys():
        table[i].append(gs.cv_results[key][i])

header = gs.cv_results.keys()
print(tabulate(table, header, tablefmt="rst"))

print()

for key, val in gs.cv_results.items():
    print('{:<20}'.format("'" + key + "':"), end='')
    if isinstance(val[0], float):
        print([float('{:.2f}'.format(f)) for f in val])
    else:
        print(val)
