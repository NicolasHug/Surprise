"""
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .algo_base import AlgoBase


class BaselineOnly(AlgoBase):
    """Algorithm predicting the baseline estimate for given user and item.

    :math:`\hat{r}_{ui} = b_{ui} = \mu + b_u + b_i`

    If user :math:`u` is unknown, then the bias :math:`b_u` is assumed to be
    zero. The same applies for item :math:`i` with :math:`b_i`.

    See section 2.1 of :cite:`Koren:2010` for details.

    Args:
        bsl_options(dict): A dictionary of options for the baseline estimates
            computation. See :ref:`baseline_estimates_configuration` for
            accepted options.
        verbose(bool): Whether to print trace messages of bias estimation,
            similarity, etc.  Default is True.
    """

    def __init__(self, bsl_options={}, verbose=True):

        AlgoBase.__init__(self, bsl_options=bsl_options)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.bu, self.bi = self.compute_baselines()

        return self

    def estimate(self, u, i):

        est = self.trainset.global_mean
        if self.trainset.knows_user(u):
            est += self.bu[u]
        if self.trainset.knows_item(i):
            est += self.bi[i]

        return est
