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

    See paper *Factor in the Neighbors: Scalable and Accurate Collaborative
    Filtering* by Yehuda Koren for details.

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

        est = self.global_mean
        #TODO: do something here (x0, user...)
        if self.trainset.knows_user(x0):
            est += self.x_biases[x0]
        if self.trainset.knows_item(y0):
            est += self.y_biases[y0]

        return est
