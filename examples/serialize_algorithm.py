"""
This module illustrates the use of the dump and load methods for serializing an
algorithm. The SVD algorithm is trained on a dataset and then serialized. It is
then reloaded and can be used again for making predictions.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

from surprise import SVD
from surprise import Dataset
from surprise import dump


data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()

algo = SVD()
algo.fit(trainset)

# Compute predictions of the 'original' algorithm.
predictions = algo.test(trainset.build_testset())

# Dump algorithm and reload it.
file_name = os.path.expanduser('~/dump_file')
dump.dump(file_name, algo=algo)
_, loaded_algo = dump.load(file_name)

# We now ensure that the algo is still the same by checking the predictions.
predictions_loaded_algo = loaded_algo.test(trainset.build_testset())
assert predictions == predictions_loaded_algo
print('Predictions are the same')
