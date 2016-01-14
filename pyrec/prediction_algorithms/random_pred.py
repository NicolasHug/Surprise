"""
Algorithm predicting a random rating.
"""

import numpy as np

from .bases import AlgoBase


class NormalPredictor(AlgoBase):
    """Algorithm predicting a random rating based on the distribution of the
    training set, which is assumed to be normal:

    :math:`\hat{r}_{ui} \sim \mathcal{N}(\hat{\mu}, \hat{\sigma})` where
    :math:`\hat{\mu}` and :math:`\hat{\sigma}` are estimated from the training data.
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.infos['name'] = 'random'

    def train(self, trainset):
        super().train(trainset)

        num = sum((r - self.global_mean)**2 for (_, _, r) in self.all_ratings)
        denum = self.n_ratings - 1  # unbiased
        self.var = num / denum

    def estimate(self, *_):

        return np.random.normal(self.global_mean, self.var)
