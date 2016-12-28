"""
Algorithm predicting only the baseline.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .algo_base import AlgoBase


class BaselineOnly(AlgoBase):
    """Algorithm predicting the baseline estimate for given user and item.

    :math:`\hat{r}_{ui} = b_{ui} = \mu + b_u + b_i`

    If user :math:`u` is unknown, then the bias :math:`b_u` is assumed to be
    zero. The same applies for item :math:`i` with :math:`b_u`.

    See section 2.1 of :cite:`Koren:2010` for details.

    Args:
        bsl_options(dict): A dictionary of options for the baseline estimates
            computation. See :ref:`baseline_estimates_configuration` for
            accepted options.

    """

    def __init__(self, bsl_options={}):

        AlgoBase.__init__(self, bsl_options=bsl_options)

    def train(self, trainset):

        AlgoBase.train(self, trainset)
        self.bu, self.bi = self.compute_baselines()

    def estimate(self, u, i):

        est = self.trainset.global_mean
        if self.trainset.knows_user(u):
            est += self.bu[u]
        if self.trainset.knows_item(i):
            est += self.bi[i]

        return est
