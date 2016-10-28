"""
the :mod:`knns` module includes some k-NN inspired algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .predictions import PredictionImpossible
from .algo_base import AlgoBase


# Important note: as soon as the algorithm uses a similarity measure, it should
# also allow the bsl_options parameter because of the pearson_baseline
# similarity. It can be done explicitely (e.g. KNNBaseline), or implicetely
# using kwargs (e.g. KNNBasic).

class KNNBasic(AlgoBase):
    """Basic collaborative filtering algorithm.

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
        k(int): The number of neighbors to take into account for aggregation.
            Pease read :ref:`this note <actual_k_note>`.
        sim_options(dict): A dictionary of options for the similarity
            measure. See :ref:`similarity_measures_configuration` for accepted
            options.
    """

    def __init__(self, k=40, sim_options={}, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.k = k

    def train(self, trainset):

        AlgoBase.train(self, trainset)
        self.compute_similarities()

    def estimate(self, x0, y0):

        #TODO: do something here (x0, user...)
        if not (self.trainset.knows_user(x0) and self.trainset.knows_item(y0)):
            raise PredictionImpossible('User and/or item is unkown.')

        neighbors = [(x, self.sim[x0, x], r) for (x, r) in self.yr[y0]]

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (_, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        try:
            est = sum_ratings / sum_sim
        except ZeroDivisionError:
            raise PredictionImpossible('Not enough neighbors.')

        details = {'actual_k' : actual_k}
        return est, details


class KNNWithMeans(AlgoBase):
    """Basic collaborative filtering algorithm, taking into account the mean
    ratings of each user.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \mu_u + \\frac{
        \\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v) \cdot (r_{vi} - \mu_v)}
        {\\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v)}

    or

    .. math::
        \hat{r}_{ui} = \mu_i + \\frac{
        \\sum\\limits_{j \in N^k_u(i)} \\text{sim}(i, j) \cdot (r_{uj} - \mu_j)}
        {\\sum\\limits_{j \in N^k_u(i)} \\text{sim}(i, j)}

    depending on the ``user_based`` field of the ``sim_options`` parameter.


    Args:
        k(int): The number of neighbors to take into account for aggregation.
            Pease read :ref:`this note <actual_k_note>`.
        sim_options(dict): A dictionary of options for the similarity
            measure. See :ref:`similarity_measures_configuration` for accepted
            options.
    """

    def __init__(self, k=40, sim_options={}, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)

        self.k = k

    def train(self, trainset):

        AlgoBase.train(self, trainset)
        self.compute_similarities()

        self.means = np.zeros(self.n_x)
        for x, ratings in self.xr.items():
            self.means[x] = np.mean([r for (_, r) in ratings])


    def estimate(self, x0, y0):

        if not (self.trainset.knows_user(x0) and self.trainset.knows_item(y0)):
            raise PredictionImpossible('User and/or item is unkown.')

        neighbors = [(x, self.sim[x0, x], r) for (x, r) in self.yr[y0]]

        est = self.means[x0]

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.means[nb])
                actual_k += 1


        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k' : actual_k}
        return est, details


class KNNBaseline(AlgoBase):
    """Basic collaborative filtering algorithm taking into account a
    *baseline* rating.


    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = b_{ui} + \\frac{
        \\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v) \cdot (r_{vi} - b_{vi})}
        {\\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v)}

    or


    .. math::
        \hat{r}_{ui} = b_{ui} + \\frac{
        \\sum\\limits_{j \in N^k_u(i)} \\text{sim}(i, j) \cdot (r_{uj} - b_{uj})}
        {\\sum\\limits_{j \in N^k_u(j)} \\text{sim}(i, j)}

    depending on the ``user_based`` field of the ``sim_options`` parameter.

    For details, see paper `Factor in the Neighbors: Scalable and Accurate
    Collaborative Filtering
    <http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf>`_  by
    Yehuda Koren.

    Args:
        k(int): The number of neighbors to take into account for aggregation.
            Pease read :ref:`this note <actual_k_note>`.
        sim_options(dict): A dictionary of options for the similarity
            measure. See :ref:`similarity_measures_configuration` for accepted
            options.
        bsl_options(dict): A dictionary of options for the baseline estimates
            computation. See :ref:`baseline_estimates_configuration` for
            accepted options.

    """

    def __init__(self, k=40, sim_options={}, bsl_options={}):

        AlgoBase.__init__(self, sim_options=sim_options,
                          bsl_options=bsl_options)

        self.k = k

    def train(self, trainset):

        AlgoBase.train(self, trainset)
        self.compute_baselines()
        self.compute_similarities()

    def estimate(self, x0, y0):

        if not (self.trainset.knows_user(x0) and self.trainset.knows_item(y0)):
            raise PredictionImpossible('User and/or item is unkown.')

        est = self.get_baseline(x0, y0)

        neighbors = [(x, self.sim[x0, x], r) for (x, r) in self.yr[y0]]

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.get_baseline(nb, y0))
                actual_k += 1

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # just baseline again

        details = {'actual_k' : actual_k}
        return est, details
