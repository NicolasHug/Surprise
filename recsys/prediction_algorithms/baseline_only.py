"""
Algorithm predicting only the baseline.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .algo_base import AlgoBase


class BaselineOnly(AlgoBase):
    """Algorithm predicting the baseline estimate for given user and item.

    :math:`\hat{r}_{ui} = b_{ui}`

    (see paper *Factor in the Neighbors: Scalable and
    Accurate Collaborative Filtering* by Yehuda Koren for details).

    Args:
        bsl_options(dict): A dictionary of options for the baseline estimates
            computation. See :ref:`baseline_estimates_configuration` for
            accepted options.
    """

    def __init__(self, bsl_options={}):

        AlgoBase.__init__(self, bsl_options=bsl_options)

    def train(self, trainset):

        AlgoBase.train(self, trainset)
        self.compute_baselines()

    def estimate(self, x0, y0):

        return self.get_baseline(x0, y0)
