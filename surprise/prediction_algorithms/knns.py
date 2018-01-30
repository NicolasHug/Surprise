"""
the :mod:`knns` module includes some k-NN inspired algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
from six import iteritems
import heapq
from collections import defaultdict

from .. import similarities as sims
from .predictions import PredictionImpossible
from .algo_base import AlgoBase


# Important note: as soon as an algorithm uses a similarity measure, it should
# also allow the bsl_options parameter because of the pearson_baseline
# similarity. It can be done explicitely (e.g. KNNBaseline), or implicitly
# using kwargs (e.g. KNNBasic).

class SymmetricAlgo(AlgoBase):
    """This is an abstract class aimed to ease the use of symmetric algorithms.

    A symmetric algorithm is an algorithm that can can be based on users or on
    items indifferently, e.g. all the algorithms in this module.

    When the algo is user-based x denotes a user and y an item. Else, it's
    reversed.

    Args:
        sim_options(dict, optional): Parameters for similarity metrics,
            used to define nearest nighbours.
            See :ref:`similarity-measure-configuration` for usage.
        fix_k_neighbors(boolean): Defines whether k neighbours used to
            predict similarity are fixed, i.e. same neighbours are used for any
            item, or k neighbours are obtained for each item as the closest
            neighbours among those who actually rated this item (for user_based
            similarity) or among all similar items (for item-item similarity).
            Default = False
            - For item-item kNN - usual strategy is to obtain k
            closest items, among all items rated by current user (use default)
            - For user_based kNN - to obtain some kind of prediction for
            a greater number of items use non-fixed neighbors (default).
            To obtain consistent ranking among top-N predctions - use fixed k.
    """

    def __init__(self, sim_options={}, fix_k_neighbors=False, **kwargs):

        AlgoBase.__init__(self, **kwargs)

        self.sim_options = sim_options
        if 'user_based' not in self.sim_options:
            self.sim_options['user_based'] = True
        self.fix_k_neighbors = fix_k_neighbors

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        return self

    def switch(self, u_stuff, i_stuff):
        """Return x_stuff and y_stuff depending on the user_based field."""

        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff

    def select_k_neighbors(self):

        k_neighbors = defaultdict(list)

        n = self.sim.shape[1]
        # find indexes of k+1 closest neighbors (as one of them is the element
        # itself)
        k_plus_neighbors = (np.argpartition(self.sim, n - self.k - 1)
                            [:, n - self.k - 1:])

        for i, neighbors in enumerate(k_plus_neighbors):
            k_neighbors[i] = [(j, self.sim[i, j]) for j in neighbors if j != i]

        self.kNN = k_neighbors

    def compute_similarities(self):
        """Build the similarity matrix.

        The way the similarity matrix is computed depends on the
        ``sim_options`` parameter passed at the creation of the algorithm (see
        :ref:`similarity_measures_configuration`).

        This method is only relevant for algorithms using a similarity measure,
        such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.

        Returns:
            The similarity matrix."""

        construction_func = {'cosine': sims.cosine, 'msd': sims.msd,
                             'pearson': sims.pearson,
                             'pearson_baseline': sims.pearson_baseline}

        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        min_support = self.sim_options.get('min_support', 1)

        args = [n_x, yr, min_support]

        name = self.sim_options.get('name', 'msd').lower()
        if name == 'pearson_baseline':
            shrinkage = self.sim_options.get('shrinkage', 100)
            bu, bi = self.compute_baselines()
            if self.sim_options['user_based']:
                bx, by = bu, bi
            else:
                bx, by = bi, bu

            args += [self.trainset.global_mean, bx, by, shrinkage]

        try:
            print('Computing the {0} similarity matrix...'.format(name))
            sim = construction_func[name](*args)
            print('Done computing similarity matrix.')
            return sim
        except KeyError:
            raise NameError('Wrong sim name ' + name + '. Allowed values ' +
                            'are ' + ', '.join(construction_func.keys()) + '.')

    def get_neighbors(self, iid, k):
        """Return the ``k`` nearest neighbors of ``iid``, which is the inner id
        of a user or an item, depending on the ``user_based`` field of
        ``sim_options`` (see :ref:`similarity_measures_configuration`).

        As the similarities are computed on the basis of a similarity measure,
        this method is only relevant for algorithms using a similarity measure,
        such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.

        For a usage example, see the :ref:`FAQ <get_k_nearest_neighbors>`.

        Args:
            iid(int): The (inner) id of the user (or item) for which we want
                the nearest neighbors. See :ref:`this note<raw_inner_note>`.

            k(int): The number of neighbors to retrieve.

        Returns:
            The list of the ``k`` (inner) ids of the closest users (or items)
            to ``iid``.
        """

        if self.sim_options['user_based']:
            all_instances = self.trainset.all_users
        else:
            all_instances = self.trainset.all_items

        others = [(x, self.sim[iid, x]) for x in all_instances() if x != iid]
        others.sort(key=lambda tple: tple[1], reverse=True)
        k_nearest_neighbors = [j for (j, _) in others[:k]]

        return k_nearest_neighbors


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
        fix_k_neighbors(boolean): Defines whether k neighbours used to
            predict similarity are fixed, i.e. same neighbours are used for any
            item, or k neighbours are obtained for each item as the closest
            neighbours among those who actually rated this item (for user_based
            similarity) or among all similar items (for item-item similarity).
            Default = False
            - For item-item kNN - usual strategy is to obtain k
            closest items, among all items rated by current user (use default)
            - For user_based kNN - to obtain some kind of prediction for
            a greater number of items use non-fixed neighbors (default).
            To obtain consistent ranking among top-N predctions - use fixed k.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, fix_k_neighbors=False,
                 **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               fix_k_neighbors=fix_k_neighbors, **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()

        if self.fix_k_neighbors:
            self.select_k_neighbors()
            del self.sim

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        if self.fix_k_neighbors:
            k_neighbors = []
            x, y = self.switch(u, i)
            for (x2, r) in self.yr[y]:
                for t in self.kNN[x]:
                    if x2 == t[0]:
                        k_neighbors.append((t[1], r))
        else:
            x, y = self.switch(u, i)
            neighbors = [(self.sim[x, x2], r) for (x2, r) in self.yr[y]]
            k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (sim, r) in k_neighbors:
            if sim > 0 and r != 0:
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
        fix_k_neighbors(boolean): Defines whether k neighbours used to
            predict similarity are fixed, i.e. same neighbours are used for any
            item, or k neighbours are obtained for each item as the closest
            neighbours among those who actually rated this item (for user_based
            similarity) or among all similar items (for item-item similarity).
            Default = False
            - For item-item kNN - usual strategy is to obtain k
            closest items, among all items rated by current user (use default)
            - For user_based kNN - to obtain some kind of prediction for
            a greater number of items use non-fixed neighbors (default).
            To obtain consistent ranking among top-N predctions - use fixed k.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, fix_k_neighbors=False,
                 **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               fix_k_neighbors=fix_k_neighbors, **kwargs)

        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()

        self.means = np.zeros(self.n_x)
        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
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

    depending on the ``user_based`` field of the ``sim_options`` parameter. For
    the best predictions, use the :func:`pearson_baseline
    <surprise.similarities.pearson_baseline>` similarity measure.

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
            options. It is recommended to use the :func:`pearson_baseline
            <surprise.similarities.pearson_baseline>` similarity measure.

        bsl_options(dict): A dictionary of options for the baseline estimates
            computation. See :ref:`baseline_estimates_configuration` for
            accepted options.
        fix_k_neighbors(boolean): Defines whether k neighbours used to
            predict similarity are fixed, i.e. same neighbours are used for any
            item, or k neighbours are obtained for each item as the closest
            neighbours among those who actually rated this item (for user_based
            similarity) or among all similar items (for item-item similarity).
            Default = False
            - For item-item kNN - usual strategy is to obtain k
            closest items, among all items rated by current user (use default)
            - For user_based kNN - to obtain some kind of prediction for
            a greater number of items use non-fixed neighbors (default).
            To obtain consistent ranking among top-N predctions - use fixed k.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, bsl_options={},
                 fix_k_neighbors=False):

        SymmetricAlgo.__init__(self, sim_options=sim_options,
                               bsl_options=bsl_options,
                               fix_k_neighbors=fix_k_neighbors)

        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.bu, self.bi = self.compute_baselines()
        self.bx, self.by = self.switch(self.bu, self.bi)
        self.sim = self.compute_similarities()

        return self

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
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
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


class KNNWithZScore(SymmetricAlgo):
    """A basic collaborative filtering algorithm, taking into account
        the z-score normalization of each user.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \mu_u + \sigma_u \\frac{ \\sum\\limits_{v \in N^k_i(u)}
        \\text{sim}(u, v) \cdot (r_{vi} - \mu_v) / \sigma_v} {\\sum\\limits_{v
        \in N^k_i(u)} \\text{sim}(u, v)}

    or

    .. math::
        \hat{r}_{ui} = \mu_i + \sigma_i \\frac{ \\sum\\limits_{j \in N^k_u(i)}
        \\text{sim}(i, j) \cdot (r_{uj} - \mu_j) / \sigma_j} {\\sum\\limits_{j
        \in N^k_u(i)} \\text{sim}(i, j)}

    depending on the ``user_based`` field of the ``sim_options`` parameter.

    If :math:`\sigma` is 0, than the overall sigma is used in that case.

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
        fix_k_neighbors(boolean): Defines whether k neighbours used to
            predict similarity are fixed, i.e. same neighbours are used for any
            item, or k neighbours are obtained for each item as the closest
            neighbours among those who actually rated this item (for user_based
            similarity) or among all similar items (for item-item similarity).
            Default = False
            - For item-item kNN - usual strategy is to obtain k
            closest items, among all items rated by current user (use default)
            - For user_based kNN - to obtain some kind of prediction for
            a greater number of items use non-fixed neighbors (default).
            To obtain consistent ranking among top-N predctions - use fixed k.
    """

    def __init__(self, k=40, min_k=1, sim_options={}, fix_k_neighbors=False,
                 **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, **kwargs,
                               fix_k_neighbors=fix_k_neighbors)

        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)

        self.means = np.zeros(self.n_x)
        self.sigmas = np.zeros(self.n_x)
        # when certain sigma is 0, use overall sigma
        self.overall_sigma = np.std(
            [r for (_, _, r) in self.trainset.all_ratings()])

        for x, ratings in iteritems(self.xr):
            self.means[x] = np.mean([r for (_, r) in ratings])
            sigma = np.std([r for (_, r) in ratings])
            self.sigmas[x] = self.overall_sigma if sigma == 0.0 else sigma

        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(x2, self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[1])

        est = self.means[x]

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (nb, sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.means[nb]) / self.sigmas[nb]
                actual_k += 1

        if actual_k < self.min_k:
            sum_ratings = 0

        try:
            est += sum_ratings / sum_sim * self.sigmas[x]
        except ZeroDivisionError:
            pass  # return mean

        details = {'actual_k': actual_k}
        return est, details
