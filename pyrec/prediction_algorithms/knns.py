"""
the :mod:`knns` module includes some some k-NN inspired algorithms.
"""

import numpy as np

from .bases import AlgoUsingSim
from .bases import AlgoWithBaseline
from .bases import PredictionImpossible


class KNNBasic(AlgoUsingSim):
    """Basic collaborative filtering algorithm.

    :math:`\hat{r}_{ui} = \\frac{
    \\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v) \cdot r_{vi}}
    {\\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v)}`
    """

    def __init__(self, training_data, item_based=False, sim_name='MSD', k=40,
                 **kwargs):
        super().__init__(training_data, item_based=item_based, sim_name=sim_name,
                 **kwargs)

        self.k = k

        self.infos['name'] = 'KNNBasic'
        self.infos['params']['similarity measure'] = sim_name
        self.infos['params']['k'] = self.k

    def estimate(self, x0, y0):

        neighbors = [(x, self.sim[x0, x], r) for (x, r) in self.yr[y0]]

        if not neighbors:
            raise PredictionImpossible

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        # compute weighted average
        sum_sim = sum_ratings = 0
        for (_, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r

        try:
            est = sum_ratings / sum_sim
        except ZeroDivisionError:
            raise PredictionImpossible

        return est


class KNNWithMeans(AlgoUsingSim):
    """Basic collaborative filtering algorithm, taking into account the mean
    ratings of each user.

    :math:`\hat{r}_{ui} = \mu_u + \\frac{
    \\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v) \cdot (r_{vi} - \mu_v)}
    {\\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v)}`
    """

    def __init__(self, training_data, item_based=False, sim_name='MSD', k=40,
                 **kwargs):
        super().__init__(training_data, item_based=item_based, sim_name=sim_name)

        self.k = k

        self.infos['name'] = 'basicWithMeans'
        self.infos['params']['similarity measure'] = sim_name
        self.infos['params']['k'] = self.k

        self.means = np.zeros(self.n_x)
        for x, ratings in self.xr.items():
            self.means[x] = np.mean([r for (_, r) in ratings])

    def estimate(self, x0, y0):

        neighbors = [(x, self.sim[x0, x], r) for (x, r) in self.yr[y0]]

        est = self.means[x0]

        if not neighbors:
            return est  # result will be just the mean of x0

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        # compute weighted average
        sum_sim = sum_ratings = 0
        for (nb, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.means[nb])

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # return mean

        return est


class KNNBaseline(AlgoWithBaseline, AlgoUsingSim):
    """Basic collaborative filtering algorithm taking into account a
    *baseline* rating (see paper *Factor in the Neighbors: Scalable and
    Accurate Collaborative Filtering* by Yehuda Koren for details).

    :math:`\hat{r}_{ui} = b_{ui} + \\frac{
    \\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v) \cdot (r_{vi} - b_{vi})}
    {\\sum\\limits_{v \in N^k_i(u)} \\text{sim}(u, v)}`
    """

    def __init__(self, training_data, item_based=False, method='als',
                 sim_name='MSD', k=40, **kwargs):
        super().__init__(training_data, item_based, method=method,
                         sim_name=sim_name, **kwargs)

        self.k = k
        self.infos['name'] = 'neighborhoodWithBaseline'
        self.infos['params']['k'] = self.k

    def estimate(self, x0, y0):

        est = self.get_baseline(x0, y0)

        neighbors = [(x, self.sim[x0, x], r) for (x, r) in self.yr[y0]]

        if not neighbors:
            return est  # result will be just the baseline

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        # compute weighted average
        sum_sim = sum_ratings = 0
        for (nb, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r - self.get_baseline(nb, y0))

        try:
            est += sum_ratings / sum_sim
        except ZeroDivisionError:
            pass  # just baseline again

        return est
