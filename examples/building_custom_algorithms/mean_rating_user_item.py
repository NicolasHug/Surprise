"""
This module descibes how to build your own prediction algorithm. Please refer
to User Guide for more insight.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from surprise import AlgoBase
from surprise import Dataset
from surprise.model_selection import cross_validate


class MyOwnAlgorithm(AlgoBase):

    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def estimate(self, u, i):

        sum_means = self.trainset.global_mean
        div = 1

        if self.trainset.knows_user(u):
            sum_means += np.mean([r for (_, r) in self.trainset.ur[u]])
            div += 1
        if self.trainset.knows_item(i):
            sum_means += np.mean([r for (_, r) in self.trainset.ir[i]])
            div += 1

        return sum_means / div


data = Dataset.load_builtin('ml-100k')
algo = MyOwnAlgorithm()

cross_validate(algo, data, verbose=True)
