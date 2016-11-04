""" Algorithm predicting a random rating.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .algo_base import AlgoBase


class NormalPredictor(AlgoBase):
    """Algorithm predicting a random rating based on the distribution of the
    training set, which is assumed to be normal.

    The prediction :math:`\hat{r}_{ui}` is generated from a normal distribution
    :math:`\mathcal{N}(\hat{\mu}, \hat{\sigma}^2)` where :math:`\hat{\mu}` and
    :math:`\hat{\sigma}` are estimated from the training data using Maximum
    Likelihood Estimation:

    .. math::
        \\hat{\mu} &= \\frac{1}{|R_{train}|} \\sum_{r_{ui} \\in R_{train}}
        r_{ui}\\\\\\\\\
        \\hat{\sigma} &= \\sqrt{\\sum_{r_{ui} \\in R_{train}}
        \\frac{(r_{ui} - \\hat{\mu})^2}{|R_{train}|}}
    """

    def __init__(self):

        AlgoBase.__init__(self)

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        num = sum((r - self.trainset.global_mean)**2
                  for (_, _, r) in self.trainset.all_ratings())
        denum = self.trainset.n_ratings
        self.sigma = np.sqrt(num / denum)

    def estimate(self, *_):

        return np.random.normal(self.trainset.global_mean, self.sigma)
