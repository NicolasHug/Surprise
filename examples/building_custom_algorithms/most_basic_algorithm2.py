"""
This module descibes how to build your own prediction algorithm. Please refer
to User Guide for more insight.
"""

from recsys import AlgoBase
from recsys import Dataset
from recsys import evaluate

from statistics import mean

class MyOwnAlgorithm(AlgoBase):

    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def train(self, trainset):

        # Here again: call base method before doing anything.
        AlgoBase.train(self, trainset)

        # Compute the average rating.
        self.the_mean = mean(r for r in self.trainset.rm.values())

    def estimate(self, u, i):

        return self.the_mean


data = Dataset.load_builtin('ml-100k')
algo = MyOwnAlgorithm()

evaluate(algo, data)

