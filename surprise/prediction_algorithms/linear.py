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
        \hat{r}_{ui} = \alpha_1 + \alpha_2^\top y_u + \alpha_3^\top z_i +
        \alpha_4^\top \text{vec}(y_u \otimes z_i)

    where :math:`\alpha_1 \in \mathbb{R}, \alpha_2 \in \mathbb{R}^o, \alpha_3
    \in \mathbb{R}^p` and :math:`\alpha_4 \in \mathbb{R}^{op}` are coefficient
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
        uf_labels = self.trainset.user_features_labels
        if_labels = self.trainset.item_features_labels

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

        coef_labels = uf_labels + if_labels
        if self.add_interactions:
            temp = np.array([X[:, v] * X[:, j] for v in range(n_uf)
                             for j in range(n_uf, n_uf + n_if)]).T
            X = np.concatenate([X, temp], axis=1)
            temp = [coef_labels[v] + '*' + coef_labels[j] for v in range(n_uf)
                    for j in range(n_uf, n_uf + n_if)]
            coef_labels += temp

        reg = linear_model.Lasso(
            alpha=self.alpha, fit_intercept=self.fit_intercept,
            normalize=self.normalize, precompute=self.precompute,
            max_iter=self.max_iter, tol=self.tol, positive=self.positive,
            random_state=self.random_state, selection=self.selection)
        reg.fit(X, y)

        self.X = X
        self.y = y
        self.coef = reg.coef_
        self.coef_labels = coef_labels
        self.intercept = reg.intercept_

    def estimate(self, u, i, u_features, i_features):

        n_uf = self.trainset.n_user_features
        n_if = self.trainset.n_item_features

        if (len(u_features) != n_uf or
                len(i_features) != n_if):
            raise PredictionImpossible(
                'User and/or item features are missing.')

        X = np.concatenate([u_features, i_features])

        if self.add_interactions:
            temp = np.array([X[v] * X[j] for v in range(n_uf)
                             for j in range(n_uf, n_uf + n_if)])
            X = np.concatenate([X, temp])

        est = self.intercept + np.dot(X, self.coef)

        return est
