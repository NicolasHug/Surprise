"""
the :mod:`knns` module includes some k-NN inspired algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import tensorflow as tf
import scipy.sparse as sps
from tffm import TFFMClassifier, TFFMRegressor
from polylearn import FactorizationMachineRegressor

from .algo_base import AlgoBase


class FMAlgo(AlgoBase):
    """This is an abstract class aimed to reduce code redundancy for
    factoration machines.
    """

    def __init__(self, order=2, n_factors=5, input_type='dense',
                 loss_function='mse', optimizer=None, reg_all=1.,
                 use_diag=False, reweight_reg=False, init_std=0.01,
                 batch_size=-1, n_epochs=100, log_dir=None,
                 session_config=None, random_state=None, verbose=False,
                 **kwargs):

        AlgoBase.__init__(self, **kwargs)
        self.order = order
        self.n_factors = n_factors  # rank in `tffm`
        self.input_type = input_type
        self.loss_function = loss_function  # {'mse', 'loss_logistic'}
        # https://www.tensorflow.org/api_guides/python/train#Optimizers
        # tf.train.Optimizer, default: AdamOptimizer(learning_rate=0.01)
        if optimizer is None:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        else:
            self.optimizer = optimizer
        self.reg_all = reg_all  # reg in `tffm`
        self.use_diag = use_diag
        self.reweight_reg = reweight_reg
        self.init_std = init_std
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.log_dir = log_dir
        self.session_config = session_config
        self.random_state = random_state  # seed in `tffm`
        self.verbose = verbose

        if self.loss_function == 'mse':
            self.tffm_model = TFFMRegressor(
                order=self.order, rank=self.n_factors,
                input_type=self.input_type, optimizer=self.optimizer,
                reg=self.reg_all, use_diag=self.use_diag,
                reweight_reg=self.reweight_reg, init_std=self.init_std,
                batch_size=self.batch_size, n_epochs=self.n_epochs,
                log_dir=self.log_dir, session_config=self.session_config,
                seed=self.random_state, verbose=self.verbose)
        elif self.loss_function == 'loss_logistic':
            # See issue #157
            raise ValueError('loss_logistic is not supported at the moment')
            self.tffm_model = TFFMClassifier(
                order=self.order, rank=self.n_factors,
                input_type=self.input_type, optimizer=self.optimizer,
                reg=self.reg_all, use_diag=self.use_diag,
                reweight_reg=self.reweight_reg, init_std=self.init_std,
                batch_size=self.batch_size, n_epochs=self.n_epochs,
                log_dir=self.log_dir, session_config=self.session_config,
                seed=self.random_state, verbose=self.verbose)
        else:
            raise ValueError(('Unknown value {} for parameter'
                              'loss_function').format(self.loss_function))

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        if self.verbose > 0:
            self.show_progress = True
        else:
            self.show_progress = False

        return self


class FMBasic(FMAlgo):
    """A basic factorization machine algorithm. With order 2, this algorithm
    is equivalent to the biased SVD algorithm.

    This code is an interface to the `tffm` library.

    Args:
    """

    def __init__(self, order=2, n_factors=5, loss_function='mse',
                 optimizer=None, reg_all=1., use_diag=False,
                 reweight_reg=False, init_std=0.01, batch_size=-1,
                 n_epochs=100, log_dir=None, session_config=None,
                 random_state=None, verbose=False, **kwargs):

        input_type = 'sparse'

        FMAlgo.__init__(self, order=order, n_factors=n_factors,
                        input_type=input_type, loss_function=loss_function,
                        optimizer=optimizer, reg_all=reg_all,
                        use_diag=use_diag, reweight_reg=reweight_reg,
                        init_std=init_std, batch_size=batch_size,
                        n_epochs=n_epochs, log_dir=log_dir,
                        session_config=session_config,
                        random_state=random_state, verbose=verbose, **kwargs)

    def fit(self, trainset):

        FMAlgo.fit(self, trainset)
        self.fm(trainset)

        return self

    def fm(self, trainset):

        n_ratings = self.trainset.n_ratings
        n_users = self.trainset.n_users
        n_items = self.trainset.n_items

        # Construct sparse X and y
        row_ind = np.empty(2 * n_ratings, dtype=int)
        col_ind = np.empty(2 * n_ratings, dtype=int)
        data = np.ones(2 * n_ratings, dtype=bool)
        y_train = np.empty(n_ratings, dtype=float)
        nonzero_counter = 0
        rating_counter = 0
        for uid, iid, rating in self.trainset.all_ratings():
            # Add user
            row_ind[nonzero_counter] = rating_counter
            col_ind[nonzero_counter] = uid
            nonzero_counter += 1
            # Add item
            row_ind[nonzero_counter] = rating_counter
            col_ind[nonzero_counter] = n_users + iid
            nonzero_counter += 1
            # Add rating
            y_train[rating_counter] = rating
            rating_counter += 1
        X_train = sps.csr_matrix((data, (row_ind, col_ind)),
                                 shape=(n_ratings, n_users + n_items),
                                 dtype=bool)

        # `Dataset` and `Trainset` do not support sample_weight at the moment.
        self.tffm_model.fit(X_train, y_train, sample_weight=None,
                            show_progress=self.show_progress)
        self.X_train = X_train
        self.y_train = y_train

    def estimate(self, u, i, *_):

        # what happens for new user/item in predict?
        # if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
        #     raise PredictionImpossible('User and/or item is unknown.')
        n_users = self.trainset.n_users
        n_items = self.trainset.n_items

        details = {}
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            X_test = sps.csr_matrix(([1., 1.], ([0, 0], [u, n_users + i])),
                                    shape=(1, n_users + n_items), dtype=bool)
            details['knows_user'] = True
            details['knows_item'] = True
        elif self.trainset.knows_user(u):
            X_test = sps.csr_matrix(([1.], ([0], [u])),
                                    shape=(1, n_users + n_items), dtype=bool)
            details['knows_user'] = True
            details['knows_item'] = False
        elif self.trainset.knows_item(i):
            X_test = sps.csr_matrix(([1.], ([0], [n_users + i])),
                                    shape=(1, n_users + n_items), dtype=bool)
            details['knows_user'] = False
            details['knows_item'] = True
        else:
            X_test = sps.csr_matrix((1, n_users + n_items), dtype=bool)
            details['knows_user'] = False
            details['knows_item'] = False

        est = self.tffm_model.predict(X_test)[0]

        return est, details


class FMBasicPL(AlgoBase):
    """A basic factorization machine algorithm. With order 2, this algorithm
    is equivalent to the biased SVD algorithm.

    This code is an interface to the `polylearn` library.

    Args:
    """

    def __init__(self, degree=2, n_factors=5, reg_all=1., reg_alpha=1.,
                 reg_beta=1., tol=1e-6,
                 random_state=None, verbose=False, **kwargs):

        self.degree = degree
        self.n_factors = n_factors  # n_components in `polylearn`
        self.reg_all = reg_all  # not in `polylearn`
        self.reg_alpha = reg_alpha  # alpha in `polylearn`
        self.reg_beta = reg_beta  # beta in `polylearn`
        self.tol = tol
        # ...
        self.random_state = random_state
        self.verbose = verbose

        self.fm_model = FactorizationMachineRegressor(
            degree=self.degree, n_components=self.n_factors,
            alpha=self.reg_alpha, beta=self.reg_beta, tol=self.tol,
            random_state=self.random_state, verbose=self.verbose)

    def fit(self, trainset):

        FMAlgo.fit(self, trainset)
        self.fm(trainset)

        return self

    def fm(self, trainset):

        n_ratings = self.trainset.n_ratings
        n_users = self.trainset.n_users
        n_items = self.trainset.n_items

        # Construct sparse X and y
        row_ind = np.empty(2 * n_ratings, dtype=int)
        col_ind = np.empty(2 * n_ratings, dtype=int)
        data = np.ones(2 * n_ratings, dtype=bool)
        y_train = np.empty(n_ratings, dtype=float)
        nonzero_counter = 0
        rating_counter = 0
        for uid, iid, rating in self.trainset.all_ratings():
            # Add user
            row_ind[nonzero_counter] = rating_counter
            col_ind[nonzero_counter] = uid
            nonzero_counter += 1
            # Add item
            row_ind[nonzero_counter] = rating_counter
            col_ind[nonzero_counter] = n_users + iid
            nonzero_counter += 1
            # Add rating
            y_train[rating_counter] = rating
            rating_counter += 1
        X_train = sps.csc_matrix((data, (row_ind, col_ind)),
                                 shape=(n_ratings, n_users + n_items),
                                 dtype=bool)

        self.tffm_model.fit(X_train, y_train)
        self.X_train = X_train
        self.y_train = y_train

    def estimate(self, u, i, *_):

        # what happens for new user/item in predict?
        # if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
        #     raise PredictionImpossible('User and/or item is unknown.')
        n_users = self.trainset.n_users
        n_items = self.trainset.n_items

        details = {}
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            X_test = sps.csc_matrix(([1., 1.], ([0, 0], [u, n_users + i])),
                                    shape=(1, n_users + n_items), dtype=bool)
            details['knows_user'] = True
            details['knows_item'] = True
        elif self.trainset.knows_user(u):
            X_test = sps.csc_matrix(([1.], ([0], [u])),
                                    shape=(1, n_users + n_items), dtype=bool)
            details['knows_user'] = True
            details['knows_item'] = False
        elif self.trainset.knows_item(i):
            X_test = sps.csc_matrix(([1.], ([0], [n_users + i])),
                                    shape=(1, n_users + n_items), dtype=bool)
            details['knows_user'] = False
            details['knows_item'] = True
        else:
            X_test = sps.csc_matrix((1, n_users + n_items), dtype=bool)
            details['knows_user'] = False
            details['knows_item'] = False

        est = self.tffm_model.predict(X_test)[0]

        return est, details
