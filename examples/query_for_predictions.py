"""
TODO
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from recsys import KNNBasic
from recsys import Dataset
from recsys import Reader
from recsys import evaluate


# Load the movielens-100k dataset and split it into 3 folds for
# cross-validation.
data = Dataset.load_builtin('ml-100k')

trainset = data.build_full_trainset()

algo = KNNBasic()

algo.train(trainset)

ruid = 196
riid = 302

uid = trainset.raw2inner_id_users[str(ruid)]
iid = trainset.raw2inner_id_users[str(riid)]

algo.predict(uid, iid, r0=4, verbose=True)

evaluate(algo, data)
