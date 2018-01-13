from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

cimport numpy as np  # noqa
import numpy as np
from six.moves import range

from .algo_base import AlgoBase
from .predictions import PredictionImpossible
from ..utils import get_rng


class BPRMF(AlgoBase):

    def __init__(self, n_factors=100, n_epochs=20, init_mean=0,
                 init_std_dev=.1, lr_all=.05, reg_all=.0025, lr_pu=None,
                 lr_qi=None, lr_qj=None, reg_pu=.0025, reg_qi=.0025,
                 reg_qj=.00025, random_state=None, verbose=False):

        AlgoBase.__init__(self)

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.verbose = verbose

        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_qj = lr_qj if lr_qj is not None else lr_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_qj = reg_qj if reg_qj is not None else reg_all

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        self.rng = get_rng(self.random_state)

        # user factors
        cdef np.ndarray[np.double_t, ndim=2] pu
        # item factors
        cdef np.ndarray[np.double_t, ndim=2] qi

        cdef int u, i, j, f
        cdef double x_ui, x_uj, x_uij, puf, qif, qjf

        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
        cdef double lr_qj = self.lr_qj

        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_qj = self.reg_qj

        pu = self.rng.normal(self.init_mean, self.init_std_dev,
                             (trainset.n_users, self.n_factors))
        qi = self.rng.normal(self.init_mean, self.init_std_dev,
                             (trainset.n_items, self.n_factors))

        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))

            u, i, j = self.sample_uij()

            x_ui = 0  # <q_i, p_u>
            x_uj = 0  # <q_j, p_u>
            for f in range(self.n_factors):
                x_ui += qi[i, f] * pu[u, f]
                x_uj += qi[j, f] * pu[u, f]
            x_uij = x_ui - x_uj
            sig = 1 / (1 + np.exp(x_uij))  # sigmoid(-x_uij)

            # update factors
            for f in range(self.n_factors):
                puf = pu[u, f]
                qif = qi[i, f]
                qjf = qi[j, f]

                pu[u, f] += lr_pu * (sig * (qif - qjf) - reg_pu * puf)
                qi[i, f] += lr_qi * (sig * (puf) - reg_qi * qif)
                qi[j, f] += lr_qj * (sig * (-puf) - reg_qj * qjf)

        self.pu = pu
        self.qi = qi

    def sample_uij(self):

        #TODO: sub optimal in many ways. improve it.

        u = self.rng.randint(self.trainset.n_users)
        u_items = set(i for (i, _) in self.trainset.ur[u])
        i = np.random.choice(tuple(u_items))
        j = None
        while j is None:
            j = np.random.randint(self.trainset.n_items)
            if j in u_items:
                j = None

        return u, i, j


    def estimate(self, u, i):

        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)

        if known_user and known_item:
            est = np.dot(self.qi[i], self.pu[u])
        else:
            raise PredictionImpossible('User and/or item are unkown.')

        return est
