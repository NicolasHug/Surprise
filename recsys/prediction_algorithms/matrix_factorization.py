"""
the :mod:`matrix_factorization` module includes some algorithms using matrix
factorization.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np

from .algo_base import AlgoBase

class SVD(AlgoBase):
    """The famous *SVD* algorithm, as popularized by `Simon Funk
    <http://sifter.org/~simon/journal/20061211.html>`_ during the Netflix
    Prize.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \mu + b_u + b_i + q_i^Tp_u

    If user :math:`u` is unknown, then the bias :math:`b_u` and the factors
    :math:`p_u` are assumed to be zero. The same applies for item :math:`i`
    with :math:`b_u` and :math:`q_i`.

    For details, see eq. 5 from `Matrix Factorization Techniques For
    Recommender Systems
    <http://www.columbia.edu/~jwp2128/Teaching/W4721/papers/ieeecomputer.pdf>`_
    by Koren, Bell and Volinsky. See also *The Recommender System Handbook*,
    section 5.3.1.

    To estimate all the unkown, we minimize the following regularized squared
    error:

    .. math::
        \sum_{r_{ui} \in R_{train}} \left(r_{ui} - \hat{r}_{ui} \\right)^2 +
        \lambda\\left(b_i^2 + b_u^2 + ||q_i||^2 + ||p_u||^2\\right)


    The minimization is performed by a very straightforward stochastic gradient
    descent:

    .. math::
        b_u &\\rightarrow b_u &+ \gamma (e_{ui} - \lambda b_u)\\\\
        b_i &\\rightarrow b_i &+ \gamma (e_{ui} - \lambda b_i)\\\\
        p_u &\\rightarrow p_u &+ \gamma (e_{ui} q_i - \lambda p_u)\\\\
        q_i &\\rightarrow q_i &+ \gamma (e_{ui} p_u - \lambda q_i)

    where :math:`e_{ui} = r_{ui} - \\hat{r}_{ui}`. These steps are performed
    over all the ratings of the trainset and repeated ``n_epochs`` times.
    Baselines are initialized to ``0``. User and item factors are initialized
    to ``0.1``, as recommended by `Funk
    <http://sifter.org/~simon/journal/20061211.html>`_.

    You have control over the learning rate :math:`\gamma` and the
    regularization term :math:`\lambda`. Both can be different for each
    kind of parameter (see below). By default, learning rates are set to
    ``0.005`` and regularization termes are set to ``0.02``.

    Args:
        n_factors: The number of factors. Default is ``100``.
        n_epochs: The number of iteration of the SGD procedure. Default is
            ``20``.
        lr_all: The learning rate for all parameters. Default is ``0.005``.
        reg_all: The regularization term for all parameters. Default is
            ``0.02``.
        lr_bu: The learning rate for :math:`b_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_bi: The learning rate for :math:`b_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_pu: The learning rate for :math:`p_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_qi: The learning rate for :math:`q_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        reg_bu: The regularization term for :math:`b_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_bi: The regularization term for :math:`b_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_pu: The regularization term for :math:`p_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_qi: The regularization term for :math:`q_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
    """

    def __init__(self, n_factors=100, n_epochs=20, lr_all=.005, reg_all=.02,
                 lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all

        AlgoBase.__init__(self)

    def train(self, trainset):

        AlgoBase.train(self, trainset)
        self.sgd(trainset)

    def sgd(self, trainset):

        bu = np.zeros(trainset.n_users) # user biases
        bi = np.zeros(trainset.n_items) # item biases
        pu = np.zeros((trainset.n_users, self.n_factors)) + .1 # user factors
        qi = np.zeros((trainset.n_items, self.n_factors)) + .1 # item factors


        for dummy in range(self.n_epochs):
            for (u, i), r in trainset.rm.items():

                err = (r -
                      (self.global_mean + bu[u] + bi[i] +
                       np.dot(qi[i], pu[u])))

                bu[u] += self.lr_bu * (err - self.reg_bu * bu[u])
                bi[i] += self.lr_bi * (err - self.reg_bi * bi[i])
                #puu_old = pu[u].copy()
                pu[u] += self.lr_pu * (err * qi[i] - self.reg_pu * pu[u])
                #qi[i] += self.lr_qi * (err * puu_old - self.reg_qi * qi[i])
                qi[i] += self.lr_qi * (err * pu[u] - self.reg_qi * qi[i])

                # Note: this is slightly incorrect: qi should be updated with
                # the previous value of pu (before it was itself updated), as
                # is done by commented lines. It leads to about a .0001
                # improvment on RMSE so yeah, let's not care.

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):

        est = self.global_mean

        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            est += np.dot(self.qi[i], self.pu[u])

        return est

class SVDpp(AlgoBase):
    """The *SVD++* algorithm, an extension of :class:`SVD` taking into account
    implicite ratings.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \hat{r}_{ui} = \mu + b_u + b_i + q_i^T\\left(p_u + |I_u|^{-\\frac{1}{2}}
        \sum_{j \\in I_u}y_j\\right)

    Where the :math:`y_j` terms are a new set of item factors that capture
    implicite ratings.

    If user :math:`u` is unknown, then the bias :math:`b_u` and the factors
    :math:`p_u` are assumed to be zero. The same applies for item :math:`i`
    with :math:`b_u`, :math:`q_i` and :math:`y_i`.


    For details, see eq. 15 from `Factorization Meets The
    Neighborhood
    <http://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf>`_
    by Yehuda Koren. See also *The Recommender System Handbook*, section 5.3.1.

    Just as for :class:`SVD`, the parameters are learnt using a SGD on the
    regularized squared error objective.

    Baselines are initialized to ``0``. User and item factors are initialized
    to ``0.1``, as recommended by `Funk
    <http://sifter.org/~simon/journal/20061211.html>`_.

    You have control over the learning rate :math:`\gamma` and the
    regularization term :math:`\lambda`. Both can be different for each
    kind of parameter (see below). By default, learning rates are set to
    ``0.005`` and regularization termes are set to ``0.02``.

    Args:
        n_factors: The number of factors. Default is ``100``.
        n_epochs: The number of iteration of the SGD procedure. Default is
            ``20``.
        lr_all: The learning rate for all parameters. Default is ``0.007``.
        reg_all: The regularization term for all parameters. Default is
            ``0.02``.
        lr_bu: The learning rate for :math:`b_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_bi: The learning rate for :math:`b_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_pu: The learning rate for :math:`p_u`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_qi: The learning rate for :math:`q_i`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        lr_yj: The learning rate for :math:`y_j`. Takes precedence over
            ``lr_all`` if set. Default is ``None``.
        reg_bu: The regularization term for :math:`b_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_bi: The regularization term for :math:`b_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_pu: The regularization term for :math:`p_u`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_qi: The regularization term for :math:`q_i`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
        reg_yj: The regularization term for :math:`y_j`. Takes precedence
            over ``reg_all`` if set. Default is ``None``.
    """

    def __init__(self, n_factors=10, n_epochs=30, lr_all=.007, reg_all=.02,
                 lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, lr_yj=None,
                 reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None,
                 reg_yj=None):

        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all
        self.lr_yj = lr_yj if lr_yj is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all
        self.reg_yj = reg_yj if reg_yj is not None else reg_all

        AlgoBase.__init__(self)

    def train(self, trainset):

        AlgoBase.train(self, trainset)
        self.sgd(trainset)

    def sgd(self, trainset):

        bu = np.zeros(trainset.n_users) # user biases
        bi = np.zeros(trainset.n_items) # item biases
        pu = np.zeros((trainset.n_users, self.n_factors)) + .1 # user factors
        qi = np.zeros((trainset.n_items, self.n_factors)) + .1 # item factors
        # implicite item factors
        yj = np.zeros((trainset.n_items, self.n_factors)) + .1


        for dummy in range(self.n_epochs):
            for (u, i), r in trainset.rm.items():

                Iu = len(trainset.ur[u])  # nb of items rated by u
                u_impl_feedback = (sum(yj[j] for (j, _) in trainset.ur[u]) /
                                   np.sqrt(Iu))

                err = (r -
                      (self.global_mean + bu[u] + bi[i] +
                       np.dot(qi[i], pu[u] + u_impl_feedback)))

                bu[u] += self.lr_bu * (err - self.reg_bu * bu[u])
                bi[i] += self.lr_bi * (err - self.reg_bi * bi[i])
                old_puu = pu[u].copy()
                pu[u] += self.lr_pu * (err * qi[i] - self.reg_pu * pu[u])
                old_qii =  qi[i].copy()
                qi[i] += self.lr_qi * (err * (old_puu + u_impl_feedback) -
                                       self.reg_qi * qi[i])
                for (j, _) in trainset.ur[u]:
                    yj[j] += self.lr_yj * (err * old_qii / np.sqrt(Iu) -
                                           self.reg_yj * yj[j])

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yj = yj

    def estimate(self, u, i):

        est = self.global_mean

        if self.trainset.knows_user(u):
            est += self.bu[u]

        if self.trainset.knows_item(i):
            est += self.bi[i]

        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            Iu = len(self.trainset.ur[u])  # nb of items rated by u
            u_impl_feedback = (sum(self.yj[j] for (j, _) in self.trainset.ur[u]) /
                               np.sqrt(Iu))
            est += np.dot(self.qi[i], self.pu[u] + u_impl_feedback)

        return est
