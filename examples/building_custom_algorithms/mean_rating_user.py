"""
This module descibes how to build your own prediction algorithm. Please refer
to User Guide for more insight.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from recsys import AlgoBase
from recsys import Dataset
from recsys import evaluate

import numpy as np

class MyOwnAlgorithm(AlgoBase):

    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def estimate(self, u, i):

        return np.mean([r for (_, r) in self.trainset.ur[u]])


data = Dataset.load_builtin('ml-100k')
algo = MyOwnAlgorithm()

evaluate(algo, data)
