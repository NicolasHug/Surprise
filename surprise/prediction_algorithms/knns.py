"""
the :mod:`knns` module includes some k-NN inspired algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from six import iteritems

from .predictions import PredictionImpossible
from .algo_base import AlgoBase


# Important note: as soon as an algorithm uses a similarity measure, it should
# also allow the bsl_options parameter because of the pearson_baseline
# similarity. It can be done explicitely (e.g. KNNBaseline), or implicetely
# using kwargs (e.g. KNNBasic).

class SymmetricAlgo(AlgoBase):
    """This is an abstract class aimed to ease the use of symmetric algorithms.

    A symmetric algorithm is an algorithm that can can be based on users or on
    items indifferently, e.g. all the algorithms in this module.

    When the algo is user-based x denotes a user and y an item. Else, it's
    reversed.
    """

    def __init__(self, sim_options={}, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

    def switch(self, u_stuff, i_stuff):
        """Return x_stuff and y_stuff depending on the user_based field."""

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff


class KNNBasic(SymmetricAlgo):
    """A basic collaborative filtering algorithm.

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

    def __init__(self, k=40, min_k=1, sim_options={}, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k
        self.min_k = min_k

    def train(self, trainset):

        SymmetricAlgo.train(self, trainset)
        self.sim = self.compute_similarities()

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (_, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details


class KNNWithMeans(SymmetricAlgo):
    """A basic collaborative filtering algorithm, taking into account the mean
    ratings of each user.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \mu_u + \\frac{ \\sum\\limits_{v \in N^k_i(u)}
        \\text{sim}(u, v) \cdot (r_{vi} - \mu_v)} {\\sum\\limits_{v \in
        N^k_i(u)} \\text{sim}(u, v)}

    or

    .. math::
        \hat{r}_{ui} = \mu_i + \\frac{ \\sum\\limits_{j \in N^k_u(i)}
        \\text{sim}(i, j) \cdot (r_{uj} - \mu_j)} {\\sum\\limits_{j \in
        N^k_u(i)} \\text{sim}(i, j)}

    depending on the ``user_based`` field of the ``sim_options`` parameter.


    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation (see :ref:`this note <actual_k_note>`). Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the neighbor
            aggregation is set to zero (so the prediction ends up being
            equivalent to the mean :math:`\mu_u` or :math:`\mu_i`). Default is
            ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure. See :ref:`similarity_measures_configuration` for accepted
            options.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, **kwargs)

        self.k = k
        self.min_k = min_k

    def train(self, trainset):

        SymmetricAlgo.train(self, trainset)
        self.sim = self.compute_similarities()

        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.means[nb])
                actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k': actual_k}
        return est, details


class KNNBaseline(SymmetricAlgo):
    """A basic collaborative filtering algorithm taking into account a
    *baseline* rating.


    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = b_{ui} + \\frac{ \\sum\\limits_{v \in N^k_i(u)}
        \\text{sim}(u, v) \cdot (r_{vi} - b_{vi})} {\\sum\\limits_{v \in
        N^k_i(u)} \\text{sim}(u, v)}

    or


    .. math::
        \hat{r}_{ui} = b_{ui} + \\frac{ \\sum\\limits_{j \in N^k_u(i)}
        \\text{sim}(i, j) \cdot (r_{uj} - b_{uj})} {\\sum\\limits_{j \in
        N^k_u(j)} \\text{sim}(i, j)}

    depending on the ``user_based`` field of the ``sim_options`` parameter.

    This algorithm corresponds to formula (3), section 2.2 of
    :cite:`Koren:2010`.

    Args:
        k(int): The (max) number of neighbors to take into account for
            aggregation (see :ref:`this note <actual_k_note>`). Default is
            ``40``.
        min_k(int): The minimum number of neighbors to take into account for
            aggregation. If there are not enough neighbors, the neighbor
            aggregation is set to zero (so the prediction ends up being
            equivalent to the baseline). Default is ``1``.
        sim_options(dict): A dictionary of options for the similarity
            measure. See :ref:`similarity_measures_configuration` for accepted
            options.
        bsl_options(dict): A dictionary of options for the baseline estimates
            computation. See :ref:`baseline_estimates_configuration` for
            accepted options.

    """

    def __init__(self, k=40, min_k=1, sim_options={}, bsl_options={}):

        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               bsl_options=bsl_options)

        self.k = k
        self.min_k = min_k

    def train(self, trainset):

        SymmetricAlgo.train(self, trainset)
        self.bu, self.bi = self.compute_baselines()
        self.bx, self.by = self.switch(self.bu, self.bi)
        self.sim = self.compute_similarities()

    def estimate(self, u, i):

        est = self.trainset.global_mean
        if self.trainset.knows_user(u):
            est += self.bu[u]
        if self.trainset.knows_item(i):
            est += self.bi[i]

        x, y = self.switch(u, i)

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            return est

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sum_sim += sim
                nb_bsl = self.trainset.global_mean + self.bx[nb] + self.by[y]
                sum_ratings += sim * (r - nb_bsl)
                actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # just baseline again

        details = {'actual_k': actual_k}
        return est, details
