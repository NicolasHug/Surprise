"""
the :mod:`co_clustering` module includes the :class:`CoClustering` algorithm.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .algo_base import AlgoBase
from .predictions import PredictionImpossible

class CoClustering(AlgoBase):

    def __init__(self, n_cltr_u=3, n_cltr_i=3):

        AlgoBase.__init__(self)

        self.n_cltr_u = n_cltr_u
        self.n_cltr_i = n_cltr_i

        self.n_epochs = 20

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        self.user_mean = [np.mean([r for (_, r) in trainset.ur[u]])
                          for u in trainset.all_users()]
        self.item_mean = [np.mean([r for (_, r) in trainset.ir[i]])
                          for i in trainset.all_items()]

        # Assign random clusters to users and items.
        self.cltr_u = np.random.randint(self.n_cltr_u, size=trainset.n_users)
        self.cltr_i = np.random.randint(self.n_cltr_i, size=trainset.n_items)

        # Average ratings of user clusters, item clusters, and coclusters.
        self.avg_cltr_u = np.empty(self.n_cltr_u)
        self.avg_cltr_i = np.empty(self.n_cltr_i)
        self.avg_cocltr = np.empty((self.n_cltr_u, self.n_cltr_i))

        for _ in range(self.n_epochs):
            self.update_averages()
            for u in self.trainset.all_users():
                self.update_user_cltr(u)
            for i in self.trainset.all_items():
                self.update_item_cltr(i)

    def update_user_cltr(self, u):

        errors = np.zeros(self.n_cltr_u)

        for cltr in range(self.n_cltr_u):
            for i, r in self.trainset.ur[u]:
                errors[cltr] += (r - self.estimate(u, i))**2

        self.cltr_u[u] = np.argmin(errors)



    def update_item_cltr(self, i):
        errors = np.zeros(self.n_cltr_i)

        for cltr in range(self.n_cltr_i):
            for u, r in self.trainset.ir[i]:
                errors[cltr] += (r - self.estimate(u, i))**2

        self.cltr_i[i] = np.argmin(errors)

    def update_averages(self):

        count_cltr_u = np.zeros(self.n_cltr_u)
        count_cltr_i = np.zeros(self.n_cltr_i)
        count_cocltr = np.zeros((self.n_cltr_u, self.n_cltr_i))

        sum_cltr_u = np.zeros(self.n_cltr_u)
        sum_cltr_i = np.zeros(self.n_cltr_i)
        sum_cocltr = np.zeros((self.n_cltr_u, self.n_cltr_i))

        for u, i, r in self.trainset.all_ratings():
            count_cltr_u[self.cltr_u[u]] += 1
            count_cltr_i[self.cltr_i[i]] += 1
            count_cocltr[self.cltr_u[u], self.cltr_i[i]] += 1

            sum_cltr_u[self.cltr_u[u]] += r
            sum_cltr_i[self.cltr_i[i]] += r
            sum_cocltr[self.cltr_u[u], self.cltr_i[i]] += r

        for cltr in range(self.n_cltr_u):
            if count_cltr_u[cltr]:
                self.avg_cltr_u[cltr] = sum_cltr_u[cltr] / count_cltr_u[cltr]
            else:
                self.avg_cltr_u[cltr] = self.trainset.global_mean


        for cltr in range(self.n_cltr_i):
            if count_cltr_i[cltr]:
                self.avg_cltr_i[cltr] = sum_cltr_i[cltr] / count_cltr_i[cltr]
            else:
                self.avg_cltr_i[cltr] = self.trainset.global_mean

        for cltr_u in range(self.n_cltr_u):
            for cltr_i in range(self.n_cltr_i):
                if count_cocltr[cltr_u, cltr_i]:
                    self.avg_cocltr[cltr_u, cltr_i] = (
                        sum_cocltr[cltr_u, cltr_i] /
                        count_cocltr[cltr_u, cltr_i])
                else:
                    self.avg_cocltr[cltr_u, cltr_i] = self.trainset.global_mean

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')


        cltr_u = self.cltr_u[u]
        cltr_i = self.cltr_i[i]

        est = (self.avg_cocltr[cltr_u, cltr_i] +
               self.user_mean[u] - self.avg_cltr_u[cltr_u] +
               self.item_mean[i] - self.avg_cltr_i[cltr_i])

        return est
