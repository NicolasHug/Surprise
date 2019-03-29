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
        self.bu, self.bi = self.compute_baselines(self.verbose)

        return self

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if not (known_user and known_item):
            print('Warning: User and/or item is unkown')

        est = self.trainset.global_mean
        if known_user:
            est += self.bu[u]
        if known_item:
            est += self.bi[i]

        return est
