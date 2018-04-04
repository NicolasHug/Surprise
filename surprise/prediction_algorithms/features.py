"""
the :mod:`features` module includes some features-based algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from six import iteritems
import heapq
from sklearn import linear_model

from .predictions import PredictionImpossible
from .algo_base import AlgoBase


class Lasso(AlgoBase):
    """A basic linear regression algorithm.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \\frac{
        \\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v) \cdot r_{vi}}
        {\\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v)}

    or

    .. math::
        \hat{r}_{ui} = \\frac{
        \\sum\\limits_{j \in N^k_u(i)} \\text{sim}(i, j) \cdot r_{uj}}
        {\\sum\\limits_{j \in N^k_u(j)} \\text{sim}(i, j)}

    depending on the ``user_based`` field of the ``sim_options`` parameter.

    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation (see :ref:`this note <actual_k_note>`). Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the prediction is
            set the the global mean of all ratings. Default is ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure. See :ref:`similarity_measures_configuration` for accepted
            options.
    """

    def __init__(self, **kwargs):

        AlgoBase.__init__(self, **kwargs)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.lasso(trainset)

        return self

    def lasso(self, trainset):

        if (self.trainset.n_user_features == 0 or
                self.trainset.n_item_features == 0):
            raise ValueError('trainset does not contain user and/or item features.')

        n_ratings = self.trainset.n_ratings
        n_uf = self.trainset.n_user_features
        n_if = self.trainset.n_item_features
        u_features = self.trainset.u_features
        i_features = self.trainset.i_features

        X = np.empty((n_ratings, n_uf + n_if))
        y = np.empty((n_ratings,))
        for k, (uid, iid, rating) in enumerate(self.trainset.all_ratings()):
            y[k] = rating
            X[k, :n_uf] = u_features[uid]
            X[k, n_uf:] = i_features[iid]

        reg = linear_model.Lasso(alpha=0.1)
        reg.fit(X, y)

        # self.X = X
        # self.y = y
        self.coef = reg.coef_
        self.intercept = reg.intercept_

    def estimate(self, u, i):

        if not (self.trainset.has_user_features(u) and
                self.trainset.has_item_features(i)):
            raise PredictionImpossible('User and/or item features '
                                       'are unknown.')

        x = np.concatenate((self.trainset.u_features[u],
                            self.trainset.i_features[i]))

        est = self.intercept + np.dot(x, self.coef)

        return est
