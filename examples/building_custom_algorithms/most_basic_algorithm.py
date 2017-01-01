"""
This module descibes how to build your own prediction algorithm. Please refer
to User Guide for more insight.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from surprise import AlgoBase
from surprise import Dataset
from surprise import evaluate


class MyOwnAlgorithm(AlgoBase):

    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def estimate(self, u, i):

        return 3


data = Dataset.load_builtin('ml-100k')
algo = MyOwnAlgorithm()

evaluate(algo, data)
