"""
the :mod:`knns` module includes some k-NN inspired algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf
import scipy.sparse as sps
from tffm import TFFMRegressor
from polylearn import FactorizationMachineRegressor

from .algo_base import AlgoBase


class FMAlgo(AlgoBase):
    """This is an abstract class aimed to reduce code redundancy for
    factoration machines.
    """

    def __init__(self, order=2, n_factors=2, input_type='dense',
                 loss_function='mse',
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                 reg_all=0, use_diag=False, reweight_reg=False, init_std=0.01,
                 batch_size=-1, n_epochs=100, log_dir=None,
                 session_config=None, random_state=None, verbose=False,
                 **kwargs):

        AlgoBase.__init__(self, **kwargs)
        self.order = order
        self.n_factors = n_factors  # rank in `tffm`
        self.input_type = input_type
        self.loss_function = loss_function  # {'mse', 'loss_logistic'}
        # https://www.tensorflow.org/api_guides/python/train#Optimizers
        self.optimizer = optimizer
        self.reg_all = reg_all  # reg in `tffm`
        self.use_diag = use_diag
        self.reweight_reg = reweight_reg
        self.init_std = np.float32(init_std)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.log_dir = log_dir
        self.session_config = session_config
        self.random_state = random_state  # seed in `tffm`
        self.verbose = verbose

        if self.loss_function == 'mse':
            self.model = TFFMRegressor(
                order=self.order, rank=self.n_factors,
                input_type=self.input_type, optimizer=self.optimizer,
                reg=self.reg_all, use_diag=self.use_diag,
                reweight_reg=self.reweight_reg, init_std=self.init_std,
                batch_size=self.batch_size, n_epochs=self.n_epochs,
                log_dir=self.log_dir, session_config=self.session_config,
                seed=self.random_state, verbose=self.verbose)
        elif self.loss_function == 'loss_logistic':
            # See issue #157 of Surprise
            raise ValueError('loss_logistic is not supported at the moment')
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
        order : int, default: 2
            Order of corresponding polynomial model.
            All interaction from bias and linear to order will be included.
        n_factors : int, default: 2
            Number of factors in low-rank appoximation.
            This value is shared across different orders of interaction.
        loss_function : str, default: 'mse'
            'mse' is the only supported loss_function at the moment.
        optimizer : tf.train.Optimizer,
            default: AdamOptimizer(learning_rate=0.01)
            Optimization method used for training
        reg_all : float, default: 0
            Strength of L2 regularization
        use_diag : bool, default: False
            Use diagonal elements of weights matrix or not.
            In the other words, should terms like x^2 be included.
            Often reffered as a "Polynomial Network".
            Default value (False) corresponds to FM.
        reweight_reg : bool, default: False
            Use frequency of features as weights for regularization or not.
            Should be useful for very sparse data and/or small batches
        init_std : float, default: 0.01
            Amplitude of random initialization
        batch_size : int, default: -1
            Number of samples in mini-batches. Shuffled every epoch.
            Use -1 for full gradient (whole training set in each batch).
        n_epoch : int, default: 100
            Default number of epoches.
            It can be overrived by explicitly provided value in fit() method.
        log_dir : str or None, default: None
            Path for storing model stats during training. Used only if is not
            None. WARNING: If such directory already exists, it will be
            removed! You can use TensorBoard to visualize the stats:
            `tensorboard --logdir={log_dir}`
        session_config : tf.ConfigProto or None, default: None
            Additional setting passed to tf.Session object.
            Useful for CPU/GPU switching, setting number of threads and so on,
            `tf.ConfigProto(device_count={'GPU': 0})` will disable GPU (if
            enabled).
        random_state : int or None, default: None
            Random seed used at graph creating time
        verbose : int, default: False
            Level of verbosity.
            Set 1 for tensorboard info only and 2 for additional stats every
            epoch.
    """

    def __init__(self, order=2, n_factors=2, loss_function='mse',
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                 reg_all=0, use_diag=False, reweight_reg=False, init_std=0.01,
                 batch_size=-1, n_epochs=100, log_dir=None,
                 session_config=None, random_state=None, verbose=False,
                 **kwargs):

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
        self.model.fit(X_train, y_train, sample_weight=None,
                       show_progress=self.show_progress)
        self.X_train = X_train
        self.y_train = y_train

    def estimate(self, u, i, *_):

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

        est = self.model.predict(X_test)[0]

        return est, details


class FMImplicit(FMAlgo):
    """A factorization machine algorithm that uses implicit ratings. With order
    2, this algorithm correspond to an extension of the biased SVD++ algorithm.

    This code is an interface to the `tffm` library.

    Args:
        order : int, default: 2
            Order of corresponding polynomial model.
            All interaction from bias and linear to order will be included.
        n_factors : int, default: 2
            Number of factors in low-rank appoximation.
            This value is shared across different orders of interaction.
        loss_function : str, default: 'mse'
            'mse' is the only supported loss_function at the moment.
        optimizer : tf.train.Optimizer,
            default: AdamOptimizer(learning_rate=0.01)
            Optimization method used for training
        reg_all : float, default: 0
            Strength of L2 regularization
        use_diag : bool, default: False
            Use diagonal elements of weights matrix or not.
            In the other words, should terms like x^2 be included.
            Often reffered as a "Polynomial Network".
            Default value (False) corresponds to FM.
        reweight_reg : bool, default: False
            Use frequency of features as weights for regularization or not.
            Should be useful for very sparse data and/or small batches
        init_std : float, default: 0.01
            Amplitude of random initialization
        batch_size : int, default: -1
            Number of samples in mini-batches. Shuffled every epoch.
            Use -1 for full gradient (whole training set in each batch).
        n_epoch : int, default: 100
            Default number of epoches.
            It can be overrived by explicitly provided value in fit() method.
        log_dir : str or None, default: None
            Path for storing model stats during training. Used only if is not
            None. WARNING: If such directory already exists, it will be
            removed! You can use TensorBoard to visualize the stats:
            `tensorboard --logdir={log_dir}`
        session_config : tf.ConfigProto or None, default: None
            Additional setting passed to tf.Session object.
            Useful for CPU/GPU switching, setting number of threads and so on,
            `tf.ConfigProto(device_count={'GPU': 0})` will disable GPU (if
            enabled).
        random_state : int or None, default: None
            Random seed used at graph creating time
        verbose : int, default: False
            Level of verbosity.
            Set 1 for tensorboard info only and 2 for additional stats every
            epoch.
    """

    def __init__(self, order=2, n_factors=2, loss_function='mse',
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                 reg_all=0, use_diag=False, reweight_reg=False, init_std=0.01,
                 batch_size=-1, n_epochs=100, log_dir=None,
                 session_config=None, random_state=None, verbose=False,
                 **kwargs):

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
        row_ind = []
        col_ind = []
        data = []
        y_train = np.empty(n_ratings, dtype=np.float32)
        rating_counter = 0
        for uid, iid, rating in self.trainset.all_ratings():
            # Add user
            row_ind.append(rating_counter)
            col_ind.append(uid)
            data.append(1.)
            # Add item
            row_ind.append(rating_counter)
            col_ind.append(n_users + iid)
            data.append(1.)
            # Add implicit ratings
            sqrt_Iu = np.sqrt(len(self.trainset.ur[uid]))
            for iid_imp, _ in self.trainset.ur[uid]:
                row_ind.append(rating_counter)
                col_ind.append(n_users + n_items + iid_imp)
                data.append(1 / sqrt_Iu)
            # Add rating
            y_train[rating_counter] = rating
            rating_counter += 1
        X_train = sps.csr_matrix((data, (row_ind, col_ind)),
                                 shape=(n_ratings, n_users + 2 * n_items),
                                 dtype=np.float32)

        # `Dataset` and `Trainset` do not support sample_weight at the moment.
        self.model.fit(X_train, y_train, sample_weight=None,
                       show_progress=self.show_progress)
        self.X_train = X_train
        self.y_train = y_train

    def estimate(self, u, i, *_):

        n_users = self.trainset.n_users
        n_items = self.trainset.n_items

        details = {}
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            data = [1., 1.]
            row_ind = [0, 0]
            col_ind = [u, n_users + i]
            details['knows_user'] = True
            details['knows_item'] = True
        elif self.trainset.knows_user(u):
            data = [1.]
            row_ind = [0]
            col_ind = [u]
            details['knows_user'] = True
            details['knows_item'] = False
        elif self.trainset.knows_item(i):
            data = [1.]
            row_ind = [0]
            col_ind = [n_users + i]
            details['knows_user'] = False
            details['knows_item'] = True
        else:
            data = []
            row_ind = []
            col_ind = []
            details['knows_user'] = False
            details['knows_item'] = False

        if self.trainset.knows_user(u):
            sqrt_Iu = np.sqrt(len(self.trainset.ur[u]))
            for iid_imp, _ in self.trainset.ur[u]:
                row_ind.append(0)
                col_ind.append(n_users + n_items + iid_imp)
                data.append(1 / sqrt_Iu)

        if data:
            X_test = sps.csr_matrix((data, (row_ind, col_ind)),
                                    shape=(1, n_users + 2 * n_items),
                                    dtype=np.float32)
        else:
            X_test = sps.csr_matrix((1, n_users + 2 * n_items),
                                    dtype=np.float32)

        est = self.model.predict(X_test)[0]

        return est, details


class FMExplicit(FMAlgo):
    """A factorization machine algorithm that uses explicit ratings.

    This code is an interface to the `tffm` library.

    Args:
        order : int, default: 2
            Order of corresponding polynomial model.
            All interaction from bias and linear to order will be included.
        n_factors : int, default: 2
            Number of factors in low-rank appoximation.
            This value is shared across different orders of interaction.
        loss_function : str, default: 'mse'
            'mse' is the only supported loss_function at the moment.
        optimizer : tf.train.Optimizer,
            default: AdamOptimizer(learning_rate=0.01)
            Optimization method used for training
        reg_all : float, default: 0
            Strength of L2 regularization
        use_diag : bool, default: False
            Use diagonal elements of weights matrix or not.
            In the other words, should terms like x^2 be included.
            Often reffered as a "Polynomial Network".
            Default value (False) corresponds to FM.
        reweight_reg : bool, default: False
            Use frequency of features as weights for regularization or not.
            Should be useful for very sparse data and/or small batches
        init_std : float, default: 0.01
            Amplitude of random initialization
        batch_size : int, default: -1
            Number of samples in mini-batches. Shuffled every epoch.
            Use -1 for full gradient (whole training set in each batch).
        n_epoch : int, default: 100
            Default number of epoches.
            It can be overrived by explicitly provided value in fit() method.
        log_dir : str or None, default: None
            Path for storing model stats during training. Used only if is not
            None. WARNING: If such directory already exists, it will be
            removed! You can use TensorBoard to visualize the stats:
            `tensorboard --logdir={log_dir}`
        session_config : tf.ConfigProto or None, default: None
            Additional setting passed to tf.Session object.
            Useful for CPU/GPU switching, setting number of threads and so on,
            `tf.ConfigProto(device_count={'GPU': 0})` will disable GPU (if
            enabled).
        random_state : int or None, default: None
            Random seed used at graph creating time
        verbose : int, default: False
            Level of verbosity.
            Set 1 for tensorboard info only and 2 for additional stats every
            epoch.
    """

    def __init__(self, order=2, n_factors=2, loss_function='mse',
                 optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
                 reg_all=0, use_diag=False, reweight_reg=False, init_std=0.01,
                 batch_size=-1, n_epochs=100, log_dir=None,
                 session_config=None, random_state=None, verbose=False,
                 **kwargs):

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
        row_ind = []
        col_ind = []
        data = []
        y_train = np.empty(n_ratings, dtype=np.float32)
        max_value = self.trainset.rating_scale[1] + self.trainset.offset
        rating_counter = 0
        for uid, iid, rating in self.trainset.all_ratings():
            # Add user
            row_ind.append(rating_counter)
            col_ind.append(uid)
            data.append(1.)
            # Add item
            row_ind.append(rating_counter)
            col_ind.append(n_users + iid)
            data.append(1.)
            # Add explicit ratings
            sqrt_Iu = np.sqrt(len(self.trainset.ur[uid]))
            for iid_exp, rating_exp in self.trainset.ur[uid]:
                row_ind.append(rating_counter)
                col_ind.append(n_users + n_items + iid_exp)
                data.append(rating_exp / (max_value * sqrt_Iu))
            # Add rating
            y_train[rating_counter] = rating
            rating_counter += 1
        X_train = sps.csr_matrix((data, (row_ind, col_ind)),
                                 shape=(n_ratings, n_users + 2 * n_items),
                                 dtype=np.float32)

        # `Dataset` and `Trainset` do not support sample_weight at the moment.
        self.model.fit(X_train, y_train, sample_weight=None,
                       show_progress=self.show_progress)
        self.X_train = X_train
        self.y_train = y_train

    def estimate(self, u, i, *_):

        n_users = self.trainset.n_users
        n_items = self.trainset.n_items

        details = {}
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            data = [1., 1.]
            row_ind = [0, 0]
            col_ind = [u, n_users + i]
            details['knows_user'] = True
            details['knows_item'] = True
        elif self.trainset.knows_user(u):
            data = [1.]
            row_ind = [0]
            col_ind = [u]
            details['knows_user'] = True
            details['knows_item'] = False
        elif self.trainset.knows_item(i):
            data = [1.]
            row_ind = [0]
            col_ind = [n_users + i]
            details['knows_user'] = False
            details['knows_item'] = True
        else:
            data = []
            row_ind = []
            col_ind = []
            details['knows_user'] = False
            details['knows_item'] = False

        if self.trainset.knows_user(u):
            sqrt_Iu = np.sqrt(len(self.trainset.ur[u]))
            max_value = self.trainset.rating_scale[1] + self.trainset.offset
            for iid_exp, rating_exp in self.trainset.ur[u]:
                row_ind.append(0)
                col_ind.append(n_users + n_items + iid_exp)
                data.append(rating_exp / (max_value * sqrt_Iu))

        if data:
            X_test = sps.csr_matrix((data, (row_ind, col_ind)),
                                    shape=(1, n_users + 2 * n_items),
                                    dtype=np.float32)
        else:
            X_test = sps.csr_matrix((1, n_users + 2 * n_items),
                                    dtype=np.float32)

        est = self.model.predict(X_test)[0]

        return est, details


class FMBasicPL(AlgoBase):
    """A basic factorization machine algorithm. With order 2, this algorithm
    is equivalent to the biased SVD algorithm.

    This code is an interface to the `polylearn` library.

    Args:
        degree : int, default: 2
            Degree of the polynomial. Corresponds to the order of feature
            interactions captured by the model. Currently only supports
            degrees up to 3.
        n_factors : int, default: 2
            Number of basis vectors to learn, a.k.a. the dimension of the
            low-rank parametrization.
        reg_alpha : float, default: 1
            Regularization amount for linear term (if ``fit_linear=True``).
        reg_beta : float, default: 1
            Regularization amount for higher-order weights.
        tol : float, default: 1e-6
            Tolerance for the stopping condition.
        fit_lower : {'explicit'|'augment'|None}, default: 'explicit'
            Whether and how to fit lower-order, non-homogeneous terms.
            - 'explicit': fits a separate P directly for each lower order.
            - 'augment': adds the required number of dummy columns (columns
               that are 1 everywhere) in order to capture lower-order terms.
               Adds ``degree - 2`` columns if ``fit_linear`` is true, or
               ``degree - 1`` columns otherwise, to account for the linear
               term.
            - None: only learns weights for the degree given.  If
              ``degree == 3``, for example, the model will only have weights
              for third-order feature interactions.
        fit_linear : {True|False}, default: True
            Whether to fit an explicit linear term <w, x> to the model, using
            coordinate descent. If False, the model can still capture linear
            effects if ``fit_lower == 'augment'``.
        warm_start : boolean, optional, default: False
            Whether to use the existing solution, if available. Useful for
            computing regularization paths or pre-initializing the model.
        init_lambdas : {'ones'|'random_signs'}, default: 'ones'
            How to initialize the predictive weights of each learned basis. The
            lambdas are not trained; using alternate signs can theoretically
            improve performance if the kernel degree is even.  The default
            value of 'ones' matches the original formulation of factorization
            machines (Rendle, 2010).
            To use custom values for the lambdas, ``warm_start`` may be used.
        max_iter : int, optional, default: 10000
            Maximum number of passes over the dataset to perform.
        random_state : int seed, RandomState instance, or None (default)
            The seed of the pseudo random number generator to use for
            initializing the parameters.
        verbose : boolean, optional, default: False
            Whether to print debugging information.
    """

    def __init__(self, degree=2, n_factors=2, reg_alpha=1., reg_beta=1.,
                 tol=1e-6, fit_lower='explicit', fit_linear=True,
                 warm_start=False, init_lambdas='ones', max_iter=10000,
                 random_state=None, verbose=False, **kwargs):

        self.degree = degree
        self.n_factors = n_factors  # n_components in `polylearn`
        self.reg_alpha = reg_alpha  # alpha in `polylearn`
        self.reg_beta = reg_beta  # beta in `polylearn`
        self.tol = tol
        self.fit_lower = fit_lower
        self.fit_linear = fit_linear
        self.warm_start = warm_start
        self.init_lambdas = init_lambdas
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        self.model = FactorizationMachineRegressor(
            degree=self.degree, n_components=self.n_factors,
            alpha=self.reg_alpha, beta=self.reg_beta, tol=self.tol,
            fit_lower=self.fit_lower, fit_linear=self.fit_linear,
            warm_start=self.warm_start, init_lambdas=self.init_lambdas,
            max_iter=self.max_iter, random_state=self.random_state,
            verbose=self.verbose)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

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

        self.model.fit(X_train, y_train)
        self.X_train = X_train
        self.y_train = y_train

        return self

    def estimate(self, u, i, *_):

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

        est = self.model.predict(X_test)[0]

        return est, details
