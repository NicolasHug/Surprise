import numpy as np

from .bases import AlgoBase


class Random(AlgoBase):
    """predict a random rating based on the distribution of the training set,
    which is assumed to be normal"""

    def __init__(self, trainingData, **kwargs):
        super().__init__(trainingData, **kwargs)
        self.infos['name'] = 'random'

        # compute unbiased variance of ratings
        num = denum = 0
        for _, _, r in self.allRatings:
            num += (r - self.meanRatings)**2
            denum += 1
        denum -= 1

        self.var = num / denum

    def estimate(self, *_):
        return np.random.normal(self.meanRatings, self.var)
