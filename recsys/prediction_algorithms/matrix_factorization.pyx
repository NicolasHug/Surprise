"""
the :mod:`matrix_factorization` module includes some algorithms using matrix
factorization.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
cimport numpy as np

from .algo_base import AlgoBase
from ..six.moves import range

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

        # OK, let's breath. I've seen so many different implementation of this
        # algorithm that I just not sure anymore of what it should do. I've
        # implemented the version as described in the BellKor papers (RS
        # Handbook, etc.). Mymedialite also does it this way. In his post
        # however, Funk seems to implicitely say that the algo looks like this
        # (see reg below):
        # for f in range(n_factors):
        #       for _ in range(n_iter):
        #           for u, i, r in all_ratings:
        #               err = r_ui - <p[u, :f+1], q[i, :f+1]>
        #               update p[u, f]
        #               update q[i, f]
        # which is also the way https://github.com/aaw/IncrementalSVD.jl
        # implemented it.
        #
        # Funk: "Anyway, this will train one feature (aspect), and in
        # particular will find the most prominent feature remaining (the one
        # that will most reduce the error that's left over after previously
        # trained features have done their best). When it's as good as it's
        # going to get, shift it onto the pile of done features, and start a
        # new one. For efficiency's sake, cache the residuals (all 100 million
        # of them) so when you're training feature 72 you don't have to wait
        # for predictRating() to re-compute the contributions of the previous
        # 71 features. You will need 2 Gig of ram, a C compiler, and good
        # programming habits to do this."

        # A note on cythonization: I haven't dived into the details, but
        # accessing 2D arrays like pu using just one of the indices like pu[u]
        # is not efficient. That's why the old (cleaner) version can't be used
        # anymore, we need to compute the dot products by hand, and update
        # user and items factors by iterating over all factors...

        # user biases
        cdef np.ndarray[np.double_t] bu = np.zeros(trainset.n_users, np.double)
        # item biases
        cdef np.ndarray[np.double_t] bi = np.zeros(trainset.n_items, np.double)
        # user factors
        cdef np.ndarray[np.double_t, ndim=2] pu = (
            np.zeros((trainset.n_users, self.n_factors), np.double) + .1)
        # item factors
        cdef np.ndarray[np.double_t, ndim=2] qi = (
            np.zeros((trainset.n_items, self.n_factors), np.double) + .1)


        cdef int u = 0
        cdef int i = 0
        cdef double r = 0
        cdef double global_mean = self.trainset.global_mean
        cdef double err = 0

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi

        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi

        cdef int f = 0
        cdef double dot = 0
        cdef double puf = 0
        cdef double qif = 0

        for _ in range(self.n_epochs):
            print(_)
            for u, i, r in trainset.all_ratings():

                # compute current error
                dot = 0  # <q_i, p_u>
                for f in range(self.n_factors):
                    dot += qi[i, f] * pu[u, f]
                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                bu[u] += lr_bu * (err - reg_bu * bu[u])
                bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * puf - reg_qi * qif)

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi

    def estimate(self, u, i):
        # Should we cythonize this as well?

        est = self.trainset.global_mean

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
        n_factors: The number of factors. Default is ``20``.
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

    def __init__(self, n_factors=20, n_epochs=20, lr_all=.007, reg_all=.02,
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

        # user biases
        cdef np.ndarray[np.double_t] bu = np.zeros(trainset.n_users, np.double)
        # item biases
        cdef np.ndarray[np.double_t] bi = np.zeros(trainset.n_items, np.double)
        # user factors
        cdef np.ndarray[np.double_t, ndim=2] pu = (
            np.zeros((trainset.n_users, self.n_factors), np.double) + .1)
        # item factors
        cdef np.ndarray[np.double_t, ndim=2] qi = (
            np.zeros((trainset.n_items, self.n_factors), np.double) + .1)
        # item implicite factors
        cdef np.ndarray[np.double_t, ndim=2] yj = (
            np.zeros((trainset.n_items, self.n_factors), np.double) + .1)


        cdef int u = 0
        cdef int i = 0
        cdef int j = 0
        cdef double r = 0
        cdef double global_mean = self.trainset.global_mean
        cdef double err = 0
        cdef np.ndarray[np.double_t] u_impl_fdb = (
            np.zeros(self.n_factors, np.double))

        cdef double lr_bu = self.lr_bu
        cdef double lr_bi = self.lr_bi
        cdef double lr_pu = self.lr_pu
        cdef double lr_qi = self.lr_qi
        cdef double lr_yj = self.lr_yj

        cdef double reg_bu = self.reg_bu
        cdef double reg_bi = self.reg_bi
        cdef double reg_pu = self.reg_pu
        cdef double reg_qi = self.reg_qi
        cdef double reg_yj = self.reg_yj

        cdef int f = 0
        cdef double dot = 0
        cdef double puf = 0
        cdef double qif = 0
        cdef double sqrt_Iu = 0
        cdef double _ = 0


        for dummy in range(self.n_epochs):
            print(dummy)
            for u, i, r in trainset.all_ratings():

                # items rated by u. This is COSTLY
                Iu = [j for (j, _) in trainset.ur[u]]
                sqrt_Iu = np.sqrt(len(Iu))

                # compute user implicite feedback
                u_impl_fdb = np.zeros(self.n_factors, np.double)
                for j in Iu:
                    for f in range(self.n_factors):
                        u_impl_fdb[f] += yj[j, f] / sqrt_Iu

                # compute current error
                dot = 0  # <q_i, (p_u + sum_{j in Iu} y_j / sqrt{Iu}>
                for f in range(self.n_factors):
                    dot += qi[i, f] * (pu[u, f] + u_impl_fdb[f])

                err = r - (global_mean + bu[u] + bi[i] + dot)

                # update biases
                bu[u] += lr_bu * (err - reg_bu * bu[u])
                bi[i] += lr_bi * (err - reg_bi * bi[i])

                # update factors
                for f in range(self.n_factors):
                    puf = pu[u, f]
                    qif = qi[i, f]
                    pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
                    qi[i, f] += lr_qi * (err * (puf + u_impl_fdb[f]) -
                                         reg_qi * qif)
                    for j in Iu:
                        yj[j, f] += lr_yj * (err * qif / sqrt_Iu -
                                             reg_yj * yj[j, f])

        self.bu = bu
        self.bi = bi
        self.pu = pu
        self.qi = qi
        self.yj = yj

    def estimate(self, u, i):

        est = self.trainset.global_mean

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
