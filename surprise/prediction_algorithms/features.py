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

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute=False, max_iter=1000, tol=0.0001, positive=False,
                 random_state=None, selection='cyclic', **kwargs):

        AlgoBase.__init__(self, **kwargs)
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.max_iter = max_iter
        self.tol = tol
        self.positive = positive
        self.random_state = random_state
        self.selection = selection

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self.lasso(trainset)

        return self

    def lasso(self, trainset):

        if (self.trainset.n_user_features == 0 or
                self.trainset.n_item_features == 0):
            raise ValueError('trainset does not contain user and/or item '
                             'features.')

        n_ratings = self.trainset.n_ratings
        n_uf = self.trainset.n_user_features
        n_if = self.trainset.n_item_features
        u_features = self.trainset.u_features
        i_features = self.trainset.i_features

        X = np.empty((n_ratings, n_uf + n_if))
        y = np.empty((n_ratings,))
        for k, (uid, iid, rating) in enumerate(self.trainset.all_ratings()):
            y[k] = rating
            try:
                X[k, :n_uf] = u_features[uid]
            except KeyError:
                raise KeyError('No features for user ' +
                               str(self.trainset.to_raw_uid(uid)))
            try:
                X[k, n_uf:] = i_features[iid]
            except KeyError:
                raise KeyError('No features for item ' +
                               str(self.trainset.to_raw_iid(iid)))

        reg = linear_model.Lasso(
            alpha=self.alpha, fit_intercept=self.fit_intercept,
            normalize=self.normalize, precompute=self.precompute,
            max_iter=self.max_iter, tol=self.tol, positive=self.positive,
            random_state=self.random_state, selection=self.selection)
        reg.fit(X, y)

        # self.X = X
        # self.y = y
        self.coef = reg.coef_
        self.intercept = reg.intercept_

    def estimate(self, u, i, u_features, i_features):

        features = np.concatenate([u_features, i_features])

        if (u_features is None or
                i_features is None or
                len(features) != len(self.coef)):
            raise PredictionImpossible('User and/or item features '
                                       'are missing.')

        est = self.intercept + np.dot(features, self.coef)

        return est
