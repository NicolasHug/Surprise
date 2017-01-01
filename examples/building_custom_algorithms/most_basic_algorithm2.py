"""
This module descibes how to build your own prediction algorithm. Please refer
to User Guide for more insight.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from surprise import AlgoBase
from surprise import Dataset
from surprise import evaluate


class MyOwnAlgorithm(AlgoBase):

    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

        # Compute the average rating. We might as well use the
        # trainset.global_mean attribute ;)
        self.the_mean = np.mean([r for (_, _, r) in
                                 self.trainset.all_ratings()])

    def estimate(self, u, i):

        return self.the_mean


data = Dataset.load_builtin('ml-100k')
algo = MyOwnAlgorithm()

evaluate(algo, data)
