"""
the :mod:`linear` module includes linear features-based algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from sklearn import linear_model

from .predictions import PredictionImpossible
from .algo_base import AlgoBase


class Lasso(AlgoBase):
    """A basic lasso algorithm with user-item interaction terms.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \alpha_0 + \alpha_1^\top y_u + \alpha_2^\top z_i +
        \alpha_3^\top \text{vec}(y_u \otimes z_i)

    where :math:`\alpha_0 \in \mathbb{R}, \alpha_1 \in \mathbb{R}^o, \alpha_2
    \in \mathbb{R}^p` and :math:`\alpha_3 \in \mathbb{R}^{op}` are coefficient
    vectors, and :math:`\otimes` represent the Kronecker product of two vectors
    (i.e., all possible cross-product combinations).

    Args:
        add_interactions(bool): Whether to add user-item interaction terms.
            Optional, default is True.
        other args: See ``sklearn`` documentation for ``linear_model.Lasso``.
    """

    def __init__(self, add_interactions=True, alpha=1.0, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000, tol=0.0001,
                 positive=False, random_state=None, selection='cyclic',
                 **kwargs):

        AlgoBase.__init__(self, **kwargs)
        self.add_interactions = add_interactions
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
                raise ValueError('No features for user ' +
                                 str(self.trainset.to_raw_uid(uid)))
            try:
                X[k, n_uf:] = i_features[iid]
            except KeyError:
                raise ValueError('No features for item ' +
                                 str(self.trainset.to_raw_iid(iid)))

        if self.add_interactions:
            X = self.add_interactions(X)

        reg = linear_model.Lasso(
            alpha=self.alpha, fit_intercept=self.fit_intercept,
            normalize=self.normalize, precompute=self.precompute,
            max_iter=self.max_iter, tol=self.tol, positive=self.positive,
            random_state=self.random_state, selection=self.selection)
        reg.fit(X, y)

        self.X = X
        self.y = y
        self.coef = reg.coef_
        self.intercept = reg.intercept_

    def add_interactions(self, X):

        n_uf = self.trainset.n_user_features
        n_if = self.trainset.n_item_features

        temp = np.array([X[:, u] * X[:, i] for u in range(n_uf)
                        for i in range(n_uf, n_uf + n_if)]).T
        X = np.concatenate([X, temp], axis=1)

        return X

    def estimate(self, u, i, u_features, i_features):

        if (u_features is None or
                len(u_features) != self.trainset.n_user_features):
            raise PredictionImpossible('User features are missing.')

        if (i_features is None or
                len(i_features) != self.trainset.n_item_features):
            raise PredictionImpossible('Item features are missing.')

        X = np.concatenate([u_features, i_features])

        if self.add_interactions:
            X = self.add_interactions(X)

        est = self.intercept + np.dot(X, self.coef)

        return est
