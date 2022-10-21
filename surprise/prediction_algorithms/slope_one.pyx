"""
the :mod:`slope_one` module includes the :class:`SlopeOne` algorithm.
"""




cimport numpy as np  # noqa
import numpy as np

from .algo_base import AlgoBase
from .predictions import PredictionImpossible


class SlopeOne(AlgoBase):
    """A simple yet accurate collaborative filtering algorithm.

    This is a straightforward implementation of the SlopeOne algorithm
    :cite:`lemire2007a`.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \\hat{r}_{ui} = \\mu_u + \\frac{1}{
        |R_i(u)|}
        \\sum\\limits_{j \\in R_i(u)} \\text{dev}(i, j),

    where :math:`R_i(u)` is the set of relevant items, i.e. the set of items
    :math:`j` rated by :math:`u` that also have at least one common user with
    :math:`i`. :math:`\\text{dev}_(i, j)` is defined as the average difference
    between the ratings of :math:`i` and those of :math:`j`:

    .. math::
        \\text{dev}(i, j) = \\frac{1}{
        |U_{ij}|}\\sum\\limits_{u \\in U_{ij}} r_{ui} - r_{uj}
    """

    def __init__(self):

        AlgoBase.__init__(self)

    def fit(self, trainset):

        cdef int n_items = trainset.n_items

        # Number of users having rated items i and j: |U_ij|
        cdef long [:, ::1] freq = np.zeros((trainset.n_items, trainset.n_items), np.int_)
        # Deviation from item i to item j: mean(r_ui - r_uj for u in U_ij)
        cdef double [:, ::1] dev = np.zeros((trainset.n_items, trainset.n_items), np.double)
        cdef int u, i, j, r_ui, r_uj

        AlgoBase.fit(self, trainset)

        # Computation of freq and dev arrays.
        for u, u_ratings in trainset.ur.items():
            for i, r_ui in u_ratings:
                for j, r_uj in u_ratings:
                    freq[i, j] += 1
                    dev[i, j] += r_ui - r_uj

        for i in range(n_items):
            dev[i, i] = 0
            for j in range(i + 1, n_items):
                dev[i, j] /= freq[i, j]
                dev[j, i] = -dev[i, j]

        self.freq = np.asarray(freq)
        self.dev = np.asarray(dev)

        # mean ratings of all users: mu_u
        self.user_mean = [np.mean([r for (_, r) in trainset.ur[u]])
                          for u in trainset.all_users()]

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unknown.')

        # Ri: relevant items for i. This is the set of items j rated by u that
        # also have common users with i (i.e. at least one user has rated both
        # i and j).
        Ri = [j for (j, _) in self.trainset.ur[u] if self.freq[i, j] > 0]
        est = self.user_mean[u]
        if Ri:
            est += sum(self.dev[i, j] for j in Ri) / len(Ri)

        return est
