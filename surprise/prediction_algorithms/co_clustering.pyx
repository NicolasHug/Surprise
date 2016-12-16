"""
the :mod:`co_clustering` module includes the :class:`CoClustering` algorithm.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
cimport numpy as np

from .algo_base import AlgoBase
from .predictions import PredictionImpossible

class CoClustering(AlgoBase):

    def __init__(self, n_cltr_u=3, n_cltr_i=3, n_epochs=20):

        AlgoBase.__init__(self)

        self.n_cltr_u = n_cltr_u
        self.n_cltr_i = n_cltr_i
        self.n_epochs = n_epochs

    def train(self, trainset):

        AlgoBase.train(self, trainset)

        cdef np.ndarray[np.double_t] user_mean
        cdef np.ndarray[np.double_t] item_mean

        cdef np.ndarray[np.int_t] cltr_u
        cdef np.ndarray[np.int_t] cltr_i

        cdef u, i, new_uc, new_ic

        cltr_u = np.random.randint(self.n_cltr_u, size=trainset.n_users)
        cltr_i = np.random.randint(self.n_cltr_i, size=trainset.n_items)

        user_mean = np.zeros(self.trainset.n_users, np.double)
        item_mean = np.zeros(self.trainset.n_items, np.double)
        for u in trainset.all_users():
            user_mean[u] = np.mean([r for (_, r) in trainset.ur[u]])
        for i in trainset.all_items():
            item_mean[i] = np.mean([r for (_, r) in trainset.ir[i]])

        self.user_mean = user_mean
        self.item_mean = item_mean
        self.cltr_u = cltr_u
        self.cltr_i = cltr_i

        for _ in range(self.n_epochs):
            print(_)
            self.update_averages()
            for u in self.trainset.all_users():
                new_cu = self.update_user_cltr(u)
                self.cltr_u[u] = new_cu
            for i in self.trainset.all_items():
                new_ci = self.update_item_cltr(i)
                self.cltr_i[i] = new_cu

        print('done training')

    def update_user_cltr(self, int u):

        cdef np.ndarray[np.double_t] errors
        cdef int uc, i, r
        cdef double est

        errors = np.zeros(self.n_cltr_u, np.double)

        for uc in range(self.n_cltr_u):
            for i, r in self.trainset.ur[u]:
                est = self.estimate(u, i)
                errors[uc] += (r - est)**2

        return np.argmin(errors)

    def update_item_cltr(self, int i):

        cdef np.ndarray[np.double_t] errors
        cdef int ic, u, r
        cdef double est

        errors = np.zeros(self.n_cltr_i, np.double)

        for ic in range(self.n_cltr_i):
            for u, r in self.trainset.ir[i]:
                est = self.estimate(u, i)
                errors[ic] += (r - est)**2

        self.cltr_i[i] = np.argmin(errors)

    def update_averages(self):

        cdef np.ndarray[np.int_t] count_cltr_u
        cdef np.ndarray[np.int_t] count_cltr_i
        cdef np.ndarray[np.int_t, ndim=2] count_cocltr

        cdef np.ndarray[np.int_t] sum_cltr_u
        cdef np.ndarray[np.int_t] sum_cltr_i
        cdef np.ndarray[np.int_t, ndim=2] sum_cocltr

        cdef np.ndarray[np.double_t] avg_cltr_u
        cdef np.ndarray[np.double_t] avg_cltr_i
        cdef np.ndarray[np.double_t, ndim=2] avg_cocltr

        cdef np.ndarray[np.int_t] cltr_u
        cdef np.ndarray[np.int_t] cltr_i

        cdef int u, i, r, uc, ic 

        count_cltr_u = np.zeros(self.n_cltr_u, np.int)
        count_cltr_i = np.zeros(self.n_cltr_i, np.int)
        count_cocltr = np.zeros((self.n_cltr_u, self.n_cltr_i), np.int)

        sum_cltr_u = np.zeros(self.n_cltr_u, np.int)
        sum_cltr_i = np.zeros(self.n_cltr_i, np.int)
        sum_cocltr = np.zeros((self.n_cltr_u, self.n_cltr_i), np.int)

        avg_cltr_u = np.zeros(self.n_cltr_u, np.double)
        avg_cltr_i = np.zeros(self.n_cltr_i, np.double)
        avg_cocltr = np.zeros((self.n_cltr_u, self.n_cltr_i), np.double)

        cltr_u = self.cltr_u
        cltr_i = self.cltr_i

        for u, i, r in self.trainset.all_ratings():
            uc = cltr_u[u]
            ic = cltr_i[i]

            count_cltr_u[uc] += 1
            count_cltr_i[ic] += 1
            count_cocltr[uc, ic] += 1

            sum_cltr_u[uc] += r
            sum_cltr_i[ic] += r
            sum_cocltr[uc, ic] += r

        for uc in range(self.n_cltr_u):
            if count_cltr_u[uc]:
                avg_cltr_u[uc] = sum_cltr_u[uc] / count_cltr_u[uc]
            else:
                avg_cltr_u[uc] = self.trainset.global_mean


        for ic in range(self.n_cltr_i):
            if count_cltr_i[ic]:
                avg_cltr_i[ic] = sum_cltr_i[ic] / count_cltr_i[ic]
            else:
                avg_cltr_i[ic] = self.trainset.global_mean

        for uc in range(self.n_cltr_u):
            for ic in range(self.n_cltr_i):
                if count_cocltr[uc, ic]:
                    avg_cocltr[uc, ic] = (sum_cocltr[uc, ic] /
                                          count_cocltr[uc, ic])
                else:
                    avg_cocltr[uc, ic] = self.trainset.global_mean

        self.avg_cltr_u = avg_cltr_u
        self.avg_cltr_i = avg_cltr_i
        self.avg_cocltr = avg_cocltr

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')


        cdef int uc = self.cltr_u[u]
        cdef int ic = self.cltr_i[i]

        est = (self.avg_cocltr[uc, ic] +
               self.user_mean[u] - self.avg_cltr_u[uc] +
               self.item_mean[i] - self.avg_cltr_i[ic])

        return est
