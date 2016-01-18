from collections import defaultdict
from itertools import combinations
import numpy as np

from .bases import AlgoBase
from .bases import PredictionImpossible
import similarities as sims

#TODO: update methods and constructor signatures


class AlgoUsingMeanDiff(AlgoBase):
    """Astract class for algorithms using the mean difference between the
    ratings of two users/items"""

    def __init__(self, user_based=True, **kwargs):

        super().__init__(user_based=user_based, **kwargs)

    def train(self, trainset):

        super().train(trainset)
        self.mean_diff = sims.compute_mean_diff(self.n_x, self.yr)


class CloneBruteforce(AlgoBase):
    """Algo based on cloning, quite rough and straightforward:
    pred(r_xy) = mean(r_x'y + k) for all x' that are k-clone of x
    """

    def __init__(self, user_based=True, **kwargs):
        super().__init__(user_based=user_based, **kwargs)

        self.infos['name'] = 'AlgoClonBruteForce'

    def train(self, trainset):

        super().train(trainset)

    def isClone(self, ra, rb, k):
        """ return True if xa (with ratings ra) is a k-clone of xb (with
        ratings rb)

            condition is sum(|(r_ai - r_bi) - k|) <= |I_ab| where |I_ab| is the
            number of common ys
        """
        diffs = [xai - xbi for (xai, xbi) in zip(ra, rb)]
        sigma = sum(abs(diff - k) for diff in diffs)
        return sigma <= len(diffs)

    def estimate(self, x0, y0):

        candidates = []
        for (x, rx) in self.yr[y0]:  # for ALL xs that have rated y0
            # find common ratings between x0 and x
            commonRatings = [(self.rm[x0, y], self.rm[x, y])
                             for (y, _) in self.xr[x0] if self.rm[x, y] > 0]
            if not commonRatings:
                continue
            ra, rb = zip(*commonRatings)
            for k in range(-4, 5):
                if self.isClone(ra, rb, k):
                    candidates.append(rx + k)

        if candidates:
            est = np.mean(candidates)
            return est
        else:
            raise PredictionImpossible


class CloneKNNMeanDiff(AlgoUsingMeanDiff):
    """Algo based on cloning:

    pred(r_xy) = av_mean(rx'y + mean_diff(x', x)) for all x' that are "close" to
    x. the term "close" can take into account constant differences other than 0
    using an appropriate similarity measure
    """

    def __init__(self, user_based=True, sim_name='MSDClone', k=40, **kwargs):

        super().__init__(user_based=user_based, sim_name=sim_name)

        self.k = k
        self.infos['name'] = 'CloneKNNMeanDiff'
        self.infos['params']['similarity measure'] = sim_name
        self.infos['params']['k'] = self.k

    def train(self, trainset):

        super().train(trainset)

    def estimate(self, x0, y0):

        neighbors = [(x, self.sim[x0, x], r) for (x, r) in self.yr[y0]]

        if not neighbors:
            raise PredictionImpossible

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        # compute weighted average
        sum_sim = sum_ratings = 0
        for (nb, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * (r + self.mean_diff[x0, nb])

        try:
            est = sum_ratings / sum_sim
        except ZeroDivisionError:
            raise PredictionImpossible

        return est
