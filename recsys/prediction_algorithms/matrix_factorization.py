"""
the :mod:`matrix_factorization` module includes some algorithms using matrix
factorization.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .bases import AlgoBase

class SVD(AlgoBase):
    """The famous *SVD* algorithm, as popularized by `Simon Funk
    <http://sifter.org/~simon/journal/20061211.html>`_ during the Netflix
    Prize.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u

    For details, see eq. 5 from `Matrix Factorization Techniques For
    Recommender Systems
    <http://www.columbia.edu/~jwp2128/Teaching/W4721/papers/ieeecomputer.pdf>`_
    by Koren, Bell and Volinsky. See also *The Recommender System Handbook*,
    section 5.3.1.

    To estimate all the unkown, we minimize the following regularized squared
    error:

    .. math::
        \sum_{r_{ui} \in R_{train}} \left(r_{ui} - (\mu + b_u + b_i +
        q_i^Tp_u)\\right)^2 + \lambda\\left(b_i^2 + b_u^2 + ||q_i||^2 +
        ||p_u||^2\\right)


    The minimization is performed by a very straightforward stochastic gradient
    descent:

    .. math::
        b_u &\\rightarrow b_u &+ \gamma (e_{ui} - \lambda b_u)\\\\
        b_i &\\rightarrow b_i &+ \gamma (e_{ui} - \lambda b_i)\\\\
        p_u &\\rightarrow p_u &+ \gamma (e_{ui} q_i - \lambda p_u)\\\\
        q_i &\\rightarrow q_i &+ \gamma (e_{ui} p_u - \lambda q_i)

    where :math:`e_{ui} = r_{ui} - \\hat{r}_{ui}`. These steps are performed
    over all the ratings of the trainset and repeated ``n_epoch`` times.
    Baselines are initialized to 0. User and item factors are initialized to
    ``0.1``, as recommended by `Funk
    <http://sifter.org/~simon/journal/20061211.html>`_.

    You have control over the learning rate :math:`\gamma` and the
    regularization parameter :math:`\lambda`. Both can be different for each
    kind of parameter (see below). By default, learning rates are set to
    ``0.005`` and regularization parameteres are set to ``0.02``.

    Args:
        n_factors: The number of factors. Default is 100.
        n_epochs: The number of iteration of the SGD procedure. Default is 20.
        lr_bu: The learning rate for :math:`b_u`. Default is ``0.005``.
        lr_bi: The learning rate for :math:`b_i`. Default is ``0.005``.
        lr_pu: The learning rate for :math:`p_u`. Default is ``0.005``.
        lr_qi: The learning rate for :math:`q_i`. Default is ``0.005``.
        reg_bu: The regularization parameter for :math:`b_u`. Default is ``0.02``.
        reg_bi: The regularization parameter for :math:`b_i`. Default is ``0.02``.
        reg_pu: The regularization parameter for :math:`p_u`. Default is ``0.02``.
        reg_qi: The regularization parameter for :math:`q_i`. Default is ``0.02``.

    """

    def __init__(self, n_factors=100, n_epochs=20,
                 lr_bu=.005, lr_bi=.005, lr_pu=.005, lr_qi=.005,
                 reg_bu=.02, reg_bi=.02, reg_pu=.02, reg_qi=.02):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.lr_pu = lr_pu
        self.lr_qi = lr_qi
        self.reg_bu = reg_bu
        self.reg_bi = reg_bi
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi

        AlgoBase.__init__(self)

    def train(self, trainset):

        AlgoBase.train(self, trainset)
        self.sgd(trainset)

    def sgd(self, trainset):

        u_biases = np.zeros(trainset.n_users)
        i_biases = np.zeros(trainset.n_items)
        u_factors = np.zeros((trainset.n_users, self.n_factors)) + .1
        i_factors = np.zeros((trainset.n_items, self.n_factors)) + .1


        for dummy in range(self.n_epochs):
            for (u, i), r in trainset.rm.items():

                err = (r -
                      (self.global_mean + u_biases[u] + i_biases[i] +
                       np.dot(u_factors[u], i_factors[i])))

                u_biases[u] += self.lr_bu * (err - self.reg_bu * u_biases[u])
                i_biases[i] += self.lr_bi * (err - self.reg_bi * i_biases[i])
                u_factors[u] += self.lr_pu * (err * i_factors[i] -
                                              self.reg_pu * u_factors[u])
                i_factors[i] += self.lr_qi * (err * u_factors[u] -
                                              self.reg_qi * i_factors[i])

        self.u_biases = u_biases
        self.i_biases = i_biases
        self.u_factors = u_factors
        self.i_factors = i_factors

    def estimate(self, u, i):

        return (self.global_mean + self.u_biases[u] + self.i_biases[i] +
                np.dot(self.u_factors[u], self.i_factors[i]))
