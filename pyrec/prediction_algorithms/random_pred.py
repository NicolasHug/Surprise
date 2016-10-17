""" Algorithm predicting a random rating.
"""

import numpy as np

from .bases import AlgoBase


class NormalPredictor(AlgoBase):
    """Algorithm predicting a random rating based on the distribution of the
    training set, which is assumed to be normal:

    :math:`\hat{r}_{ui}` is generated from a normal distribution
    :math:`\mathcal{N}(\hat{\mu}, \hat{\sigma}^2)` where :math:`\hat{\mu}` and
    :math:`\hat{\sigma}` are estimated from the training data using Maximum
    Likelihood Estimation:

    .. math::
        \\hat{\mu} &= \\frac{1}{|R_{train}|} \\sum_{r_{ui} \\in R_{train}}
        r_{ui}\\\\
        \\hat{\sigma} &= \\sqrt{\\sum_{r_{ui} \\in R_{train}}
        \\frac{(r_{ui} - \\hat{\mu})^2}{|R_{train}|}}
    """

    def __init__(self, **kwargs):

        AlgoBase.__init__(self, **kwargs)

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        num = sum((r - self.global_mean)**2 for (_, _, r) in self.all_ratings)
        denum = self.n_ratings
        self.sigma= np.sqrt(num / denum)

    def estimate(self, *_):

        #TODO: this is actually incorrect.
        # Argument should be std dev
        return np.random.normal(self.global_mean, self.sigma)
