"""
This module descibes how to build your own prediction algorithm. Please refer
to User Guide for more insight.
"""

from recsys import AlgoBase
from recsys import Dataset
from recsys import evaluate

from statistics import mean

class MyOwnAlgorithm(AlgoBase):

    def __init__(self, sim_options={}, bsl_options={}):

        AlgoBase.__init__(self, sim_options=sim_options,
                                bsl_options=bsl_options)

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        # compute baselines and similarities
        self.compute_baselines()
        self.compute_similarities()

    def estimate(self, u, i):

        return self.get_baseline(u, i)


data = Dataset.load_builtin('ml-100k')
algo = MyOwnAlgorithm()

evaluate(algo, data)

