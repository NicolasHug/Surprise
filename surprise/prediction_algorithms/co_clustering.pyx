"""
the :mod:`co_clustering` module includes the :class:`CoClustering` algorithm.
"""




cimport numpy as np  # noqa
import numpy as np

from .algo_base import AlgoBase
from ..utils import get_rng


class CoClustering(AlgoBase):
    """A collaborative filtering algorithm based on co-clustering.

    This is a straightforward implementation of :cite:`George:2005`.

    Basically, users and items are assigned some clusters :math:`C_u`,
    :math:`C_i`, and some co-clusters :math:`C_{ui}`.

    The prediction :math:`\\hat{r}_{ui}` is set as:

    .. math::
        \\hat{r}_{ui} = \\overline{C_{ui}} + (\\mu_u - \\overline{C_u}) + (\\mu_i
        - \\overline{C_i}),

    where :math:`\\overline{C_{ui}}` is the average rating of co-cluster
    :math:`C_{ui}`, :math:`\\overline{C_u}` is the average rating of
    :math:`u`'s cluster, and :math:`\\overline{C_i}` is the average rating of
    :math:`i`'s cluster. If the user is unknown, the prediction is
    :math:`\\hat{r}_{ui} = \\mu_i`. If the item is unknown, the prediction is
    :math:`\\hat{r}_{ui} = \\mu_u`. If both the user and the item are unknown,
    the prediction is :math:`\\hat{r}_{ui} = \\mu`.

    Clusters are assigned using a straightforward optimization method, much
    like k-means.

    Args:
       n_cltr_u(int): Number of user clusters. Default is ``3``.
       n_cltr_i(int): Number of item clusters. Default is ``3``.
       n_epochs(int): Number of iteration of the optimization loop. Default is
           ``20``.
       random_state(int, RandomState instance from numpy, or ``None``):
           Determines the RNG that will be used for initialization. If
           int, ``random_state`` will be used as a seed for a new RNG. This is
           useful to get the same initialization over multiple calls to
           ``fit()``.  If RandomState instance, this same instance is used as
           RNG. If ``None``, the current RNG from numpy is used.  Default is
           ``None``.
       verbose(bool): If True, the current epoch will be printed. Default is
           ``False``.

    """

    def __init__(self, n_cltr_u=3, n_cltr_i=3, n_epochs=20, random_state=None,
                 verbose=False):

        AlgoBase.__init__(self)

        self.n_cltr_u = n_cltr_u
        self.n_cltr_i = n_cltr_i
        self.n_epochs = n_epochs
        self.verbose=verbose
        self.random_state = random_state

    def fit(self, trainset):

        # All this implementation was hugely inspired from MyMediaLite:
        # https://github.com/zenogantner/MyMediaLite/blob/master/src/MyMediaLite/RatingPrediction/CoClustering.cs

        AlgoBase.fit(self, trainset)

        # User and item means
        cdef double [::1] user_mean
        cdef double [::1] item_mean

        # User and items clusters
        cdef long [::1] cltr_u
        cdef long [::1] cltr_i

        # Average rating of user clusters, item clusters and co-clusters
        cdef double [::1] avg_cltr_u
        cdef double [::1] avg_cltr_i
        cdef double [:, ::1] avg_cocltr

        cdef double [::1] errors
        cdef int u, i, uc, ic
        cdef double est, r

        # Randomly assign users and items to intial clusters
        rng = get_rng(self.random_state)
        _cltr_u = rng.randint(self.n_cltr_u, size=trainset.n_users)
        _cltr_i = rng.randint(self.n_cltr_i, size=trainset.n_items)
        cltr_u = _cltr_u.astype(np.int64)
        cltr_i = _cltr_i.astype(np.int64)

        # Compute user and item means
        user_mean = np.zeros(self.trainset.n_users, np.double)
        item_mean = np.zeros(self.trainset.n_items, np.double)
        for u in trainset.all_users():
            user_mean[u] = np.mean([r for (_, r) in trainset.ur[u]])
        for i in trainset.all_items():
            item_mean[i] = np.mean([r for (_, r) in trainset.ir[i]])

        # Optimization loop. This could be optimized a bit by checking if
        # clusters where effectively updated and early stop if they did not.
        for epoch in range(self.n_epochs):

            if self.verbose:
                print("Processing epoch {}".format(epoch))

            # Update averages of clusters
            avg_cltr_u, avg_cltr_i, avg_cocltr = compute_averages(self,
                                                                  cltr_u,
                                                                  cltr_i)
            # set user cluster to the one that minimizes squarred error of all
            # the user's ratings.
            for u in self.trainset.all_users():
                errors = np.zeros(self.n_cltr_u, np.double)
                for uc in range(self.n_cltr_u):
                    for i, r in self.trainset.ur[u]:
                        ic = cltr_i[i]
                        est = (avg_cocltr[uc, ic] +
                               user_mean[u] - avg_cltr_u[uc] +
                               item_mean[i] - avg_cltr_i[ic])
                        errors[uc] += (r - est)**2
                cltr_u[u] = np.argmin(errors)

            # set item cluster to the one that minimizes squarred error over
            # all the item's ratings.
            for i in self.trainset.all_items():
                errors = np.zeros(self.n_cltr_i, np.double)
                for ic in range(self.n_cltr_i):
                    for u, r in self.trainset.ir[i]:
                        uc = cltr_u[u]
                        est = (avg_cocltr[uc, ic] +
                               user_mean[u] - avg_cltr_u[uc] +
                               item_mean[i] - avg_cltr_i[ic])
                        errors[ic] += (r - est)**2
                cltr_i[i] = np.argmin(errors)

        # Compute averages one last time as clusters may have change
        avg_cltr_u, avg_cltr_i, avg_cocltr = compute_averages(self,
                                                              cltr_u,
                                                              cltr_i)
        # Set cdefed arrays as attributes as they are needed for prediction
        self.cltr_u = np.asarray(cltr_u)
        self.cltr_i = np.asarray(cltr_i)

        self.user_mean = np.asarray(user_mean)
        self.item_mean = np.asarray(item_mean)

        self.avg_cltr_u = np.asarray(avg_cltr_u)
        self.avg_cltr_i = np.asarray(avg_cltr_i)
        self.avg_cocltr = np.asarray(avg_cocltr)

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            return self.trainset.global_mean

        if not self.trainset.knows_user(u):
            return self.cltr_i[i]

        if not self.trainset.knows_item(i):
            return self.cltr_u[u]

        # I doubt cdefing makes any difference here as cython has no clue about
        # arrays self.stuff... But maybe?
        _u = u
        _i = i
        uc = self.cltr_u[_u]
        ic = self.cltr_i[_i]

        est = (self.avg_cocltr[uc, ic] +
               self.user_mean[_u] - self.avg_cltr_u[uc] +
               self.item_mean[_i] - self.avg_cltr_i[ic])

        return est


def compute_averages(self, cltr_u, cltr_i):
    """Compute cluster averages.

    Args:
        cltr_u: current user clusters
        cltr_i: current item clusters

    Returns:
        Three arrays: averages of user clusters, item clusters and
        co-clusters.
    """

    cdef long [::1] _cltr_u = cltr_u
    cdef long [::1] _cltr_i = cltr_i

    # Number of entities in user clusters, item clusters and co-clusters.
    cdef long [::1] count_cltr_u
    cdef long [::1] count_cltr_i
    cdef long [:, ::1] count_cocltr

    # Sum of ratings for entities in each cluster
    cdef double [::1] sum_cltr_u
    cdef double [::1] sum_cltr_i
    cdef double [:, ::1] sum_cocltr

    # The averages of each cluster (what will be returned)
    cdef double [::1] avg_cltr_u
    cdef double [::1] avg_cltr_i
    cdef double [:, ::1] avg_cocltr

    cdef int u, i, uc, ic
    cdef double r
    cdef double global_mean = self.trainset.global_mean

    # Initialize everything to zero
    count_cltr_u = np.zeros(self.n_cltr_u, np.int64)
    count_cltr_i = np.zeros(self.n_cltr_i, np.int64)
    count_cocltr = np.zeros((self.n_cltr_u, self.n_cltr_i), np.int64)

    sum_cltr_u = np.zeros(self.n_cltr_u, np.double)
    sum_cltr_i = np.zeros(self.n_cltr_i, np.double)
    sum_cocltr = np.zeros((self.n_cltr_u, self.n_cltr_i), np.double)

    avg_cltr_u = np.zeros(self.n_cltr_u, np.double)
    avg_cltr_i = np.zeros(self.n_cltr_i, np.double)
    avg_cocltr = np.zeros((self.n_cltr_u, self.n_cltr_i), np.double)

    # Compute counts and sums for every cluster.
    for u, i, r in self.trainset.all_ratings():
        uc = _cltr_u[u]
        ic = _cltr_i[i]

        count_cltr_u[uc] += 1
        count_cltr_i[ic] += 1
        count_cocltr[uc, ic] += 1

        sum_cltr_u[uc] += r
        sum_cltr_i[ic] += r
        sum_cocltr[uc, ic] += r

    # Then set the averages for users...
    for uc in range(self.n_cltr_u):
        if count_cltr_u[uc]:
            avg_cltr_u[uc] = sum_cltr_u[uc] / count_cltr_u[uc]
        else:
            avg_cltr_u[uc] = global_mean

    # ... for items
    for ic in range(self.n_cltr_i):
        if count_cltr_i[ic]:
            avg_cltr_i[ic] = sum_cltr_i[ic] / count_cltr_i[ic]
        else:
            avg_cltr_i[ic] = global_mean

    # ... and for co-clusters
    for uc in range(self.n_cltr_u):
        for ic in range(self.n_cltr_i):
            if count_cocltr[uc, ic]:
                avg_cocltr[uc, ic] = (sum_cocltr[uc, ic] /
                                      count_cocltr[uc, ic])
            else:
                avg_cocltr[uc, ic] = global_mean

    return avg_cltr_u, avg_cltr_i, avg_cocltr
