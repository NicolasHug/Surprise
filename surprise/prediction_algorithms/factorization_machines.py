"""
the :mod:`knns` module includes some k-NN inspired algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import pandas as pd
# import tensorflow as tf
import scipy.sparse as sps
import xlearn as xl
# from tffm import TFFMRegressor
# from polylearn import FactorizationMachineRegressor

from .predictions import PredictionImpossible
from .algo_base import AlgoBase


class FM(AlgoBase):
    """A factorization machine algorithm.

    This code is an interface to the `xlearn` library.

    Args:
        rating_lst (list of str or `None`): This list specifies what
            information from the `raw_ratings` to put in the `x` vector.
            Accepted list values are 'userID', 'itemID', 'imp_u_rating' and
            'exp_u_rating'. Implicit and explicit user rating values are scaled
            by the number of values. If `None`, no info is added.
        user_lst (list of str or `None`): This list specifies what
            information from the `user_features` to put in the `x` vector.
            Accepted list values consist of the names of features. If `None`,
            no info is added.
        item_lst (list of str or `None`): This list specifies what
            information from the `item_features` to put in the `x` vector.
            Accepted list values consist of the names of features. If `None`,
            no info is added.
        task : str, default: 'reg'
            'binary' or 'reg'.
        metric : str, default: 'rmse'
            'acc', 'prec', 'recall', 'f1', 'auc' for classification.
            'mae', 'mape', 'rmse', 'rmsd' for regression.
        lr : float, default: 0.2
            Learning rate for optimization method. If you choose 'adagrad'
            method, the learning rate will be changed adaptively.
        reg : float, default: 0.00002
            Strength of L2 regularization. It can be disabled by setting it to
            zero.
        k : int, default: 4
            Number of latent factors in low-rank appoximation.
        init : float, default: 0.66
            Used to initialize model.
        alpha : float, default: 0.3
            Hyper parameter for 'ftrl'.
        beta : float, default: 1.0
            Hyper parameter for 'ftrl'.
        lambda_1 : float, default : 0.00001
            Hyper parameter for 'ftrl'.
        lambda_2 : float, default : 0.00002
            Hyper parameter for 'ftrl'.
        nthread : int, default : 1
            Number of CPU cores.
        epoch : int, default : 10
            Number of epochs. This value could be changed in early-stop.
        opt : str, default : 'adagrad'
            'sgd', 'adagrad' and 'ftrl' are accepted values for the
            optimization method.
        stop_window : int, default : 2
            Size of the stop window for early-stopping.
        use_bin : bool, default : True
            Generate bin file for training and testing.
        use_norm : bool, default : True
            Use instance-wise normalization.
        use_lock_free : bool, default : True
            Use lock-free training. This does not allow reproducible results.
        use_early_stop : bool, default : True
            Use early stopping.
        random_state : int, default: 1
            Random seed used to shuffle data set.
        verbose : int, default: False
            Level of verbosity.
        modeltxtpath : 'str', default: 'model.txt'
            Path and filename of model in text format.
        modelpath : 'str', default: 'model.out'
            Path and filename of model in binary format.
    """

    def __init__(self, rating_lst=['userID', 'itemID'], user_lst=None,
                 item_lst=None, task='reg', metric='rmse', lr=0.2, reg=0.00002,
                 k=4, init=0.66, alpha=0.3, beta=1.0, lambda_1=0.00001,
                 lambda_2=0.00002, nthread=1, epoch=10, opt='adagrad',
                 stop_window=2, use_bin=True, use_norm=True,
                 use_lock_free=True, use_early_stop=True,
                 random_state=1, verbose=False, modeltxtpath='model.txt',
                 modelpath='model.out', **kwargs):

        AlgoBase.__init__(self, **kwargs)
        self.rating_lst = rating_lst
        self.user_lst = user_lst
        self.item_lst = item_lst
        self.task = task
        self.metric = metric
        self.lr = lr
        self.reg = reg
        self.k = k
        self.init = init
        self.alpha = alpha
        self.beta = beta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.nthread = nthread
        self.epoch = epoch
        self.opt = opt
        self.stop_window = stop_window
        self.use_bin = use_bin
        self.use_norm = use_norm
        self.use_lock_free = use_lock_free
        self.use_early_stop = use_early_stop
        self.random_state = random_state
        self.verbose = verbose
        self.modeltxtpath = modeltxtpath
        self.modelpath = modelpath

        self.model = xl.create_fm()
        self.param = {'task': self.task,
                      'metric': self.metric,
                      'lr': self.lr,
                      'lambda': self.reg,
                      'k': self.k,
                      'init': self.init,
                      'alpha': self.alpha,
                      'beta': self.beta,
                      'lambda_1': self.lambda_1,
                      'lambda_2': self.lambda_2,
                      'nthread': self.nthread,
                      'epoch': self.epoch,
                      'opt': self.opt,
                      'stop_window': self.stop_window}
        if not self.use_bin:
            self.model.setNoBin()
        if not self.use_norm:
            self.model.disableNorm()
        if not self.use_lock_free:
            self.model.disableLockFree()
        if not self.use_early_stop:
            self.model.disableEarlyStop()
        if not verbose:
            self.model.setQuiet()

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)
        self._construct_libsvm()

        # Fit without validation data set
        xdm_train = xl.DMatrix(
            self.libsvm_df.loc[:, self.libsvm_df.columns != 'rating'],
            self.libsvm_df.loc[:, 'rating'])
        self.model.setTrain(xdm_train)
        self.model.setTXTModel(self.modeltxtpath)
        self.model.fit(self.param, self.modelpath)

        # Fit with validation data set (TODO)

        # Load text model
        # This is used to define `self.bias`, `self.linear_coefs` and
        # `self.inter_coefs`
        self._load_txt_model()

        # Delete model files (TODO)

        return self

    def _construct_libsvm(self):
        """ Outputs the data in a libsvm format. This format is used by FM
        algorithms. It is assumed that these features are correctly encoded.
        These dummies are created (if needed) using only the info in the
        trainset.
        """

        if self.user_lst and (self.trainset.n_user_features == 0):
            raise ValueError('user_lst cannot be used since '
                             'there are no user_features')
        if self.item_lst and (self.trainset.n_item_features == 0):
            raise ValueError('item_lst cannot be used since '
                             'there are no item_features')

        n_ratings = self.trainset.n_ratings
        n_users = self.trainset.n_users
        n_items = self.trainset.n_items

        # Construct ratings_df from trainset
        # The IDs are unique and start at 0
        ratings_df = pd.DataFrame([tup for tup in self.trainset.all_ratings()],
                                  columns=['userID', 'itemID', 'rating'])

        # Initialize df with rating values
        libsvm_df = pd.DataFrame(ratings_df['rating'])

        # Add rating features
        if self.rating_lst:
            for feature in self.rating_lst:
                if feature == 'userID':
                    libsvm_df = pd.concat([libsvm_df, pd.get_dummies(
                        ratings_df['userID'], prefix='userID')], axis=1)
                elif feature == 'itemID':
                    libsvm_df = pd.concat([libsvm_df, pd.get_dummies(
                        ratings_df['itemID'], prefix='itemID')], axis=1)
                elif feature == 'imp_u_rating':
                    temp = np.zeros((n_ratings, n_items))
                    for row in ratings_df.itertuples():
                        iid = row.itemID
                        all_u_ratings = self.trainset.ur[row.userID]
                        for other_iid, rating in all_u_ratings:
                            if other_iid != iid:  # only the other ratings
                                temp[row.Index, other_iid] = 1
                    count = np.count_nonzero(temp, axis=1)[:, None]
                    count[count == 0] = 1  # remove zeros for division
                    temp = temp / count
                    cols = ['imp_u_rating_{}'.format(i)
                            for i in range(n_items)]
                    libsvm_df = pd.concat([libsvm_df, pd.DataFrame(
                        temp, columns=cols)], axis=1)
                elif feature == 'exp_u_rating':
                    # a rating is at least 1 with the offset
                    temp = np.zeros((n_ratings, n_items))
                    for row in ratings_df.itertuples():
                        iid = row.itemID
                        all_u_ratings = self.trainset.ur[row.userID]
                        for other_iid, rating in all_u_ratings:
                            if other_iid != iid:  # only the other ratings
                                temp[row.Index, other_iid] = rating
                    count = np.count_nonzero(temp, axis=1)[:, None]
                    count[count == 0] = 1  # remove zeros for division
                    temp = temp / count
                    cols = ['exp_u_rating_{}'.format(i)
                            for i in range(n_items)]
                    libsvm_df = pd.concat([libsvm_df, pd.DataFrame(
                        temp, columns=cols)], axis=1)
                else:
                    raise ValueError('{} is not an accepted value '
                                     'for rating_lst'.format(feature))

        # Add user features
        if self.user_lst:
            temp = pd.DataFrame(
                [self.trainset.u_features[uid]
                 for uid in ratings_df['userID']],
                columns=self.trainset.user_features_labels)
            for feature in self.user_lst:
                if feature in self.trainset.user_features_labels:
                    libsvm_df[feature] = temp[feature]
                else:
                    raise ValueError(
                        '{} is not part of user_features'.format(feature))

        # Add item features
        if self.item_lst:
            temp = pd.DataFrame(
                [self.trainset.i_features[iid]
                 for iid in ratings_df['itemID']],
                columns=self.trainset.item_features_labels)
            for feature in self.item_lst:
                if feature in self.trainset.item_features_labels:
                    libsvm_df[feature] = temp[feature]
                else:
                    raise ValueError(
                        '{} is not part of item_features'.format(feature))

        self.libsvm_df = libsvm_df
        self.libsvm_feature_nb = self.libsvm_df.shape[1] - 1

    def _load_txt_model(self):

        temp = pd.read_csv(self.modeltxtpath, sep=':', header=None)
        coefs = temp.loc[:, 1].tolist()
        coefs = " ".join(coefs)
        coefs = np.array(list(map(float, coefs.split())))

        self.bias = coefs[0]
        self.linear_coefs = coefs[1:self.libsvm_feature_nb + 1]
        self.inter_coefs = coefs[self.libsvm_feature_nb + 1:].reshape(
            self.libsvm_feature_nb, self.k)

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


# class FMAlgo(AlgoBase):
#     """This is an abstract class aimed to reduce code redundancy for
#     factoration machines.
#     """

#     def __init__(self, order=2, n_factors=2, input_type='dense',
#                  loss_function='mse',
#                  optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
#                  reg_all=0, use_diag=False, reweight_reg=False, init_std=0.01,
#                  batch_size=-1, n_epochs=100, log_dir=None,
#                  session_config=None, random_state=None, verbose=False,
#                  **kwargs):

#         AlgoBase.__init__(self, **kwargs)
#         self.order = order
#         self.n_factors = n_factors  # rank in `tffm`
#         self.input_type = input_type
#         self.loss_function = loss_function  # {'mse', 'loss_logistic'}
#         # https://www.tensorflow.org/api_guides/python/train#Optimizers
#         self.optimizer = optimizer
#         self.reg_all = reg_all  # reg in `tffm`
#         self.use_diag = use_diag
#         self.reweight_reg = reweight_reg
#         self.init_std = np.float32(init_std)
#         self.batch_size = batch_size
#         self.n_epochs = n_epochs
#         self.log_dir = log_dir
#         self.session_config = session_config
#         self.random_state = random_state  # seed in `tffm`
#         self.verbose = verbose

#         if self.loss_function == 'mse':
#             self.model = TFFMRegressor(
#                 order=self.order, rank=self.n_factors,
#                 input_type=self.input_type, optimizer=self.optimizer,
#                 reg=self.reg_all, use_diag=self.use_diag,
#                 reweight_reg=self.reweight_reg, init_std=self.init_std,
#                 batch_size=self.batch_size, n_epochs=self.n_epochs,
#                 log_dir=self.log_dir, session_config=self.session_config,
#                 seed=self.random_state, verbose=self.verbose)
#         elif self.loss_function == 'loss_logistic':
#             # See issue #157 of Surprise
#             raise ValueError('loss_logistic is not supported at the moment')
#         else:
#             raise ValueError(('Unknown value {} for parameter'
#                               'loss_function').format(self.loss_function))

#     def fit(self, trainset):

#         AlgoBase.fit(self, trainset)

#         if self.verbose > 0:
#             self.show_progress = True
#         else:
#             self.show_progress = False

#         return self


# class FMBasic(FMAlgo):
#     """A basic factorization machine algorithm. With order 2, this algorithm
#     is equivalent to the biased SVD algorithm.

#     This code is an interface to the `tffm` library.

#     Args:
#         order : int, default: 2
#             Order of corresponding polynomial model.
#             All interaction from bias and linear to order will be included.
#         n_factors : int, default: 2
#             Number of factors in low-rank appoximation.
#             This value is shared across different orders of interaction.
#         loss_function : str, default: 'mse'
#             'mse' is the only supported loss_function at the moment.
#         optimizer : tf.train.Optimizer,
#             default: AdamOptimizer(learning_rate=0.01)
#             Optimization method used for training
#         reg_all : float, default: 0
#             Strength of L2 regularization
#         use_diag : bool, default: False
#             Use diagonal elements of weights matrix or not.
#             In the other words, should terms like x^2 be included.
#             Often reffered as a "Polynomial Network".
#             Default value (False) corresponds to FM.
#         reweight_reg : bool, default: False
#             Use frequency of features as weights for regularization or not.
#             Should be useful for very sparse data and/or small batches
#         init_std : float, default: 0.01
#             Amplitude of random initialization
#         batch_size : int, default: -1
#             Number of samples in mini-batches. Shuffled every epoch.
#             Use -1 for full gradient (whole training set in each batch).
#         n_epoch : int, default: 100
#             Default number of epoches.
#             It can be overrived by explicitly provided value in fit() method.
#         log_dir : str or None, default: None
#             Path for storing model stats during training. Used only if is not
#             None. WARNING: If such directory already exists, it will be
#             removed! You can use TensorBoard to visualize the stats:
#             `tensorboard --logdir={log_dir}`
#         session_config : tf.ConfigProto or None, default: None
#             Additional setting passed to tf.Session object.
#             Useful for CPU/GPU switching, setting number of threads and so on,
#             `tf.ConfigProto(device_count={'GPU': 0})` will disable GPU (if
#             enabled).
#         random_state : int or None, default: None
#             Random seed used at graph creating time
#         verbose : int, default: False
#             Level of verbosity.
#             Set 1 for tensorboard info only and 2 for additional stats every
#             epoch.
#     """

#     def __init__(self, order=2, n_factors=2, loss_function='mse',
#                  optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
#                  reg_all=0, use_diag=False, reweight_reg=False, init_std=0.01,
#                  batch_size=-1, n_epochs=100, log_dir=None,
#                  session_config=None, random_state=None, verbose=False,
#                  **kwargs):

#         input_type = 'sparse'

#         FMAlgo.__init__(self, order=order, n_factors=n_factors,
#                         input_type=input_type, loss_function=loss_function,
#                         optimizer=optimizer, reg_all=reg_all,
#                         use_diag=use_diag, reweight_reg=reweight_reg,
#                         init_std=init_std, batch_size=batch_size,
#                         n_epochs=n_epochs, log_dir=log_dir,
#                         session_config=session_config,
#                         random_state=random_state, verbose=verbose, **kwargs)

#     def fit(self, trainset):

#         FMAlgo.fit(self, trainset)
#         self.fm(trainset)

#         return self

#     def fm(self, trainset):

#         n_ratings = self.trainset.n_ratings
#         n_users = self.trainset.n_users
#         n_items = self.trainset.n_items

#         # Construct sparse X and y
#         row_ind = np.empty(2 * n_ratings, dtype=int)
#         col_ind = np.empty(2 * n_ratings, dtype=int)
#         data = np.ones(2 * n_ratings, dtype=bool)
#         y_train = np.empty(n_ratings, dtype=float)
#         nonzero_counter = 0
#         rating_counter = 0
#         for uid, iid, rating in self.trainset.all_ratings():
#             # Add user
#             row_ind[nonzero_counter] = rating_counter
#             col_ind[nonzero_counter] = uid
#             nonzero_counter += 1
#             # Add item
#             row_ind[nonzero_counter] = rating_counter
#             col_ind[nonzero_counter] = n_users + iid
#             nonzero_counter += 1
#             # Add rating
#             y_train[rating_counter] = rating
#             rating_counter += 1
#         X_train = sps.csr_matrix((data, (row_ind, col_ind)),
#                                  shape=(n_ratings, n_users + n_items),
#                                  dtype=bool)

#         # `Dataset` and `Trainset` do not support sample_weight at the moment.
#         self.model.fit(X_train, y_train, sample_weight=None,
#                        show_progress=self.show_progress)
#         self.X_train = X_train
#         self.y_train = y_train

#     def estimate(self, u, i, *_):

#         n_users = self.trainset.n_users
#         n_items = self.trainset.n_items

#         details = {}
#         if self.trainset.knows_user(u) and self.trainset.knows_item(i):
#             X_test = sps.csr_matrix(([1., 1.], ([0, 0], [u, n_users + i])),
#                                     shape=(1, n_users + n_items), dtype=bool)
#             details['knows_user'] = True
#             details['knows_item'] = True
#         elif self.trainset.knows_user(u):
#             X_test = sps.csr_matrix(([1.], ([0], [u])),
#                                     shape=(1, n_users + n_items), dtype=bool)
#             details['knows_user'] = True
#             details['knows_item'] = False
#         elif self.trainset.knows_item(i):
#             X_test = sps.csr_matrix(([1.], ([0], [n_users + i])),
#                                     shape=(1, n_users + n_items), dtype=bool)
#             details['knows_user'] = False
#             details['knows_item'] = True
#         else:
#             X_test = sps.csr_matrix((1, n_users + n_items), dtype=bool)
#             details['knows_user'] = False
#             details['knows_item'] = False

#         est = self.model.predict(X_test)[0]

#         return est, details


# class FMImplicit(FMAlgo):
#     """A factorization machine algorithm that uses implicit ratings. With order
#     2, this algorithm correspond to an extension of the biased SVD++ algorithm.

#     This code is an interface to the `tffm` library.

#     Args:
#         order : int, default: 2
#             Order of corresponding polynomial model.
#             All interaction from bias and linear to order will be included.
#         n_factors : int, default: 2
#             Number of factors in low-rank appoximation.
#             This value is shared across different orders of interaction.
#         loss_function : str, default: 'mse'
#             'mse' is the only supported loss_function at the moment.
#         optimizer : tf.train.Optimizer,
#             default: AdamOptimizer(learning_rate=0.01)
#             Optimization method used for training
#         reg_all : float, default: 0
#             Strength of L2 regularization
#         use_diag : bool, default: False
#             Use diagonal elements of weights matrix or not.
#             In the other words, should terms like x^2 be included.
#             Often reffered as a "Polynomial Network".
#             Default value (False) corresponds to FM.
#         reweight_reg : bool, default: False
#             Use frequency of features as weights for regularization or not.
#             Should be useful for very sparse data and/or small batches
#         init_std : float, default: 0.01
#             Amplitude of random initialization
#         batch_size : int, default: -1
#             Number of samples in mini-batches. Shuffled every epoch.
#             Use -1 for full gradient (whole training set in each batch).
#         n_epoch : int, default: 100
#             Default number of epoches.
#             It can be overrived by explicitly provided value in fit() method.
#         log_dir : str or None, default: None
#             Path for storing model stats during training. Used only if is not
#             None. WARNING: If such directory already exists, it will be
#             removed! You can use TensorBoard to visualize the stats:
#             `tensorboard --logdir={log_dir}`
#         session_config : tf.ConfigProto or None, default: None
#             Additional setting passed to tf.Session object.
#             Useful for CPU/GPU switching, setting number of threads and so on,
#             `tf.ConfigProto(device_count={'GPU': 0})` will disable GPU (if
#             enabled).
#         random_state : int or None, default: None
#             Random seed used at graph creating time
#         verbose : int, default: False
#             Level of verbosity.
#             Set 1 for tensorboard info only and 2 for additional stats every
#             epoch.
#     """

#     def __init__(self, order=2, n_factors=2, loss_function='mse',
#                  optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
#                  reg_all=0, use_diag=False, reweight_reg=False, init_std=0.01,
#                  batch_size=-1, n_epochs=100, log_dir=None,
#                  session_config=None, random_state=None, verbose=False,
#                  **kwargs):

#         input_type = 'sparse'

#         FMAlgo.__init__(self, order=order, n_factors=n_factors,
#                         input_type=input_type, loss_function=loss_function,
#                         optimizer=optimizer, reg_all=reg_all,
#                         use_diag=use_diag, reweight_reg=reweight_reg,
#                         init_std=init_std, batch_size=batch_size,
#                         n_epochs=n_epochs, log_dir=log_dir,
#                         session_config=session_config,
#                         random_state=random_state, verbose=verbose, **kwargs)

#     def fit(self, trainset):

#         FMAlgo.fit(self, trainset)
#         self.fm(trainset)

#         return self

#     def fm(self, trainset):

#         n_ratings = self.trainset.n_ratings
#         n_users = self.trainset.n_users
#         n_items = self.trainset.n_items

#         # Construct sparse X and y
#         row_ind = []
#         col_ind = []
#         data = []
#         y_train = np.empty(n_ratings, dtype=np.float32)
#         rating_counter = 0
#         for uid, iid, rating in self.trainset.all_ratings():
#             # Add user
#             row_ind.append(rating_counter)
#             col_ind.append(uid)
#             data.append(1.)
#             # Add item
#             row_ind.append(rating_counter)
#             col_ind.append(n_users + iid)
#             data.append(1.)
#             # Add implicit ratings
#             sqrt_Iu = np.sqrt(len(self.trainset.ur[uid]))
#             for iid_imp, _ in self.trainset.ur[uid]:
#                 row_ind.append(rating_counter)
#                 col_ind.append(n_users + n_items + iid_imp)
#                 data.append(1 / sqrt_Iu)
#             # Add rating
#             y_train[rating_counter] = rating
#             rating_counter += 1
#         X_train = sps.csr_matrix((data, (row_ind, col_ind)),
#                                  shape=(n_ratings, n_users + 2 * n_items),
#                                  dtype=np.float32)

#         # `Dataset` and `Trainset` do not support sample_weight at the moment.
#         self.model.fit(X_train, y_train, sample_weight=None,
#                        show_progress=self.show_progress)
#         self.X_train = X_train
#         self.y_train = y_train

#     def estimate(self, u, i, *_):

#         n_users = self.trainset.n_users
#         n_items = self.trainset.n_items

#         details = {}
#         if self.trainset.knows_user(u) and self.trainset.knows_item(i):
#             data = [1., 1.]
#             row_ind = [0, 0]
#             col_ind = [u, n_users + i]
#             details['knows_user'] = True
#             details['knows_item'] = True
#         elif self.trainset.knows_user(u):
#             data = [1.]
#             row_ind = [0]
#             col_ind = [u]
#             details['knows_user'] = True
#             details['knows_item'] = False
#         elif self.trainset.knows_item(i):
#             data = [1.]
#             row_ind = [0]
#             col_ind = [n_users + i]
#             details['knows_user'] = False
#             details['knows_item'] = True
#         else:
#             data = []
#             row_ind = []
#             col_ind = []
#             details['knows_user'] = False
#             details['knows_item'] = False

#         if self.trainset.knows_user(u):
#             sqrt_Iu = np.sqrt(len(self.trainset.ur[u]))
#             for iid_imp, _ in self.trainset.ur[u]:
#                 row_ind.append(0)
#                 col_ind.append(n_users + n_items + iid_imp)
#                 data.append(1 / sqrt_Iu)

#         if data:
#             X_test = sps.csr_matrix((data, (row_ind, col_ind)),
#                                     shape=(1, n_users + 2 * n_items),
#                                     dtype=np.float32)
#         else:
#             X_test = sps.csr_matrix((1, n_users + 2 * n_items),
#                                     dtype=np.float32)

#         est = self.model.predict(X_test)[0]

#         return est, details


# class FMExplicit(FMAlgo):
#     """A factorization machine algorithm that uses explicit ratings.

#     This code is an interface to the `tffm` library.

#     Args:
#         order : int, default: 2
#             Order of corresponding polynomial model.
#             All interaction from bias and linear to order will be included.
#         n_factors : int, default: 2
#             Number of factors in low-rank appoximation.
#             This value is shared across different orders of interaction.
#         loss_function : str, default: 'mse'
#             'mse' is the only supported loss_function at the moment.
#         optimizer : tf.train.Optimizer,
#             default: AdamOptimizer(learning_rate=0.01)
#             Optimization method used for training
#         reg_all : float, default: 0
#             Strength of L2 regularization
#         use_diag : bool, default: False
#             Use diagonal elements of weights matrix or not.
#             In the other words, should terms like x^2 be included.
#             Often reffered as a "Polynomial Network".
#             Default value (False) corresponds to FM.
#         reweight_reg : bool, default: False
#             Use frequency of features as weights for regularization or not.
#             Should be useful for very sparse data and/or small batches
#         init_std : float, default: 0.01
#             Amplitude of random initialization
#         batch_size : int, default: -1
#             Number of samples in mini-batches. Shuffled every epoch.
#             Use -1 for full gradient (whole training set in each batch).
#         n_epoch : int, default: 100
#             Default number of epoches.
#             It can be overrived by explicitly provided value in fit() method.
#         log_dir : str or None, default: None
#             Path for storing model stats during training. Used only if is not
#             None. WARNING: If such directory already exists, it will be
#             removed! You can use TensorBoard to visualize the stats:
#             `tensorboard --logdir={log_dir}`
#         session_config : tf.ConfigProto or None, default: None
#             Additional setting passed to tf.Session object.
#             Useful for CPU/GPU switching, setting number of threads and so on,
#             `tf.ConfigProto(device_count={'GPU': 0})` will disable GPU (if
#             enabled).
#         random_state : int or None, default: None
#             Random seed used at graph creating time
#         verbose : int, default: False
#             Level of verbosity.
#             Set 1 for tensorboard info only and 2 for additional stats every
#             epoch.
#     """

#     def __init__(self, order=2, n_factors=2, loss_function='mse',
#                  optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
#                  reg_all=0, use_diag=False, reweight_reg=False, init_std=0.01,
#                  batch_size=-1, n_epochs=100, log_dir=None,
#                  session_config=None, random_state=None, verbose=False,
#                  **kwargs):

#         input_type = 'sparse'

#         FMAlgo.__init__(self, order=order, n_factors=n_factors,
#                         input_type=input_type, loss_function=loss_function,
#                         optimizer=optimizer, reg_all=reg_all,
#                         use_diag=use_diag, reweight_reg=reweight_reg,
#                         init_std=init_std, batch_size=batch_size,
#                         n_epochs=n_epochs, log_dir=log_dir,
#                         session_config=session_config,
#                         random_state=random_state, verbose=verbose, **kwargs)

#     def fit(self, trainset):

#         FMAlgo.fit(self, trainset)
#         self.fm(trainset)

#         return self

#     def fm(self, trainset):

#         n_ratings = self.trainset.n_ratings
#         n_users = self.trainset.n_users
#         n_items = self.trainset.n_items

#         # Construct sparse X and y
#         row_ind = []
#         col_ind = []
#         data = []
#         y_train = np.empty(n_ratings, dtype=np.float32)
#         max_value = self.trainset.rating_scale[1] + self.trainset.offset
#         rating_counter = 0
#         for uid, iid, rating in self.trainset.all_ratings():
#             # Add user
#             row_ind.append(rating_counter)
#             col_ind.append(uid)
#             data.append(1.)
#             # Add item
#             row_ind.append(rating_counter)
#             col_ind.append(n_users + iid)
#             data.append(1.)
#             # Add explicit ratings
#             sqrt_Iu = np.sqrt(len(self.trainset.ur[uid]))
#             for iid_exp, rating_exp in self.trainset.ur[uid]:
#                 row_ind.append(rating_counter)
#                 col_ind.append(n_users + n_items + iid_exp)
#                 data.append(rating_exp / (max_value * sqrt_Iu))
#             # Add rating
#             y_train[rating_counter] = rating
#             rating_counter += 1
#         X_train = sps.csr_matrix((data, (row_ind, col_ind)),
#                                  shape=(n_ratings, n_users + 2 * n_items),
#                                  dtype=np.float32)

#         # `Dataset` and `Trainset` do not support sample_weight at the moment.
#         self.model.fit(X_train, y_train, sample_weight=None,
#                        show_progress=self.show_progress)
#         self.X_train = X_train
#         self.y_train = y_train

#     def estimate(self, u, i, *_):

#         n_users = self.trainset.n_users
#         n_items = self.trainset.n_items

#         details = {}
#         if self.trainset.knows_user(u) and self.trainset.knows_item(i):
#             data = [1., 1.]
#             row_ind = [0, 0]
#             col_ind = [u, n_users + i]
#             details['knows_user'] = True
#             details['knows_item'] = True
#         elif self.trainset.knows_user(u):
#             data = [1.]
#             row_ind = [0]
#             col_ind = [u]
#             details['knows_user'] = True
#             details['knows_item'] = False
#         elif self.trainset.knows_item(i):
#             data = [1.]
#             row_ind = [0]
#             col_ind = [n_users + i]
#             details['knows_user'] = False
#             details['knows_item'] = True
#         else:
#             data = []
#             row_ind = []
#             col_ind = []
#             details['knows_user'] = False
#             details['knows_item'] = False

#         if self.trainset.knows_user(u):
#             sqrt_Iu = np.sqrt(len(self.trainset.ur[u]))
#             max_value = self.trainset.rating_scale[1] + self.trainset.offset
#             for iid_exp, rating_exp in self.trainset.ur[u]:
#                 row_ind.append(0)
#                 col_ind.append(n_users + n_items + iid_exp)
#                 data.append(rating_exp / (max_value * sqrt_Iu))

#         if data:
#             X_test = sps.csr_matrix((data, (row_ind, col_ind)),
#                                     shape=(1, n_users + 2 * n_items),
#                                     dtype=np.float32)
#         else:
#             X_test = sps.csr_matrix((1, n_users + 2 * n_items),
#                                     dtype=np.float32)

#         est = self.model.predict(X_test)[0]

#         return est, details


# class FMFeatures(FMAlgo):
#     """A factorization machine algorithm that uses available features.

#     WARNING: Features should be pre-scaled to an absolute value less or equal
#     to 1 if using a high value for `order`.

#     This code is an interface to the `tffm` library.

#     Args:
#         order : int, default: 2
#             Order of corresponding polynomial model.
#             All interaction from bias and linear to order will be included.
#         n_factors : int, default: 2
#             Number of factors in low-rank appoximation.
#             This value is shared across different orders of interaction.
#         loss_function : str, default: 'mse'
#             'mse' is the only supported loss_function at the moment.
#         optimizer : tf.train.Optimizer,
#             default: AdamOptimizer(learning_rate=0.01)
#             Optimization method used for training
#         reg_all : float, default: 0
#             Strength of L2 regularization
#         use_diag : bool, default: False
#             Use diagonal elements of weights matrix or not.
#             In the other words, should terms like x^2 be included.
#             Often reffered as a "Polynomial Network".
#             Default value (False) corresponds to FM.
#         reweight_reg : bool, default: False
#             Use frequency of features as weights for regularization or not.
#             Should be useful for very sparse data and/or small batches
#         init_std : float, default: 0.01
#             Amplitude of random initialization
#         batch_size : int, default: -1
#             Number of samples in mini-batches. Shuffled every epoch.
#             Use -1 for full gradient (whole training set in each batch).
#         n_epoch : int, default: 100
#             Default number of epoches.
#             It can be overrived by explicitly provided value in fit() method.
#         log_dir : str or None, default: None
#             Path for storing model stats during training. Used only if is not
#             None. WARNING: If such directory already exists, it will be
#             removed! You can use TensorBoard to visualize the stats:
#             `tensorboard --logdir={log_dir}`
#         session_config : tf.ConfigProto or None, default: None
#             Additional setting passed to tf.Session object.
#             Useful for CPU/GPU switching, setting number of threads and so on,
#             `tf.ConfigProto(device_count={'GPU': 0})` will disable GPU (if
#             enabled).
#         random_state : int or None, default: None
#             Random seed used at graph creating time
#         verbose : int, default: False
#             Level of verbosity.
#             Set 1 for tensorboard info only and 2 for additional stats every
#             epoch.
#     """

#     def __init__(self, order=2, n_factors=2, loss_function='mse',
#                  optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
#                  reg_all=0, use_diag=False, reweight_reg=False, init_std=0.01,
#                  batch_size=-1, n_epochs=100, log_dir=None,
#                  session_config=None, random_state=None, verbose=False,
#                  **kwargs):

#         input_type = 'sparse'

#         FMAlgo.__init__(self, order=order, n_factors=n_factors,
#                         input_type=input_type, loss_function=loss_function,
#                         optimizer=optimizer, reg_all=reg_all,
#                         use_diag=use_diag, reweight_reg=reweight_reg,
#                         init_std=init_std, batch_size=batch_size,
#                         n_epochs=n_epochs, log_dir=log_dir,
#                         session_config=session_config,
#                         random_state=random_state, verbose=verbose, **kwargs)

#     def fit(self, trainset):

#         FMAlgo.fit(self, trainset)
#         self.fm(trainset)

#         return self

#     def fm(self, trainset):

#         n_ratings = self.trainset.n_ratings
#         n_users = self.trainset.n_users
#         n_items = self.trainset.n_items
#         n_user_features = self.trainset.n_user_features
#         n_item_features = self.trainset.n_item_features

#         # Construct sparse X and y
#         row_ind = []
#         col_ind = []
#         data = []
#         y_train = np.empty(n_ratings, dtype=np.float32)
#         rating_counter = 0
#         for uid, iid, rating in self.trainset.all_ratings():
#             # Add user
#             row_ind.append(rating_counter)
#             col_ind.append(uid)
#             data.append(1.)
#             # Add item
#             row_ind.append(rating_counter)
#             col_ind.append(n_users + iid)
#             data.append(1.)
#             # Add user features (if any)
#             if n_user_features > 0:
#                 if self.trainset.has_user_features(uid):
#                     for n, value in enumerate(self.trainset.u_features[uid]):
#                         if value != 0:
#                             row_ind.append(rating_counter)
#                             col_ind.append(n_users + n_items + n)
#                             data.append(value)
#                 else:
#                     raise ValueError('No features for user ' +
#                                      str(self.trainset.to_raw_uid(uid)))
#             # Add item features (if any)
#             if n_item_features > 0:
#                 if self.trainset.has_item_features(iid):
#                     for n, value in enumerate(self.trainset.i_features[iid]):
#                         if value != 0:
#                             row_ind.append(rating_counter)
#                             col_ind.append(n_users + n_items +
#                                            n_user_features + n)
#                             data.append(value)
#                 else:
#                     raise ValueError('No features for item ' +
#                                      str(self.trainset.to_raw_iid(iid)))
#             # Add rating
#             y_train[rating_counter] = rating
#             rating_counter += 1
#         X_train = sps.csr_matrix((data, (row_ind, col_ind)),
#                                  shape=(n_ratings, n_users + n_items +
#                                         n_user_features + n_item_features),
#                                  dtype=np.float32)

#         # `Dataset` and `Trainset` do not support sample_weight at the moment.
#         self.model.fit(X_train, y_train, sample_weight=None,
#                        show_progress=self.show_progress)
#         self.X_train = X_train
#         self.y_train = y_train

#     def estimate(self, u, i, u_features, i_features):

#         n_users = self.trainset.n_users
#         n_items = self.trainset.n_items
#         n_user_features = self.trainset.n_user_features
#         n_item_features = self.trainset.n_item_features

#         if (len(u_features) != n_user_features or
#                 len(i_features) != n_item_features):
#             raise PredictionImpossible(
#                 'User and/or item features are missing.')

#         details = {}
#         if self.trainset.knows_user(u) and self.trainset.knows_item(i):
#             data = [1., 1.]
#             row_ind = [0, 0]
#             col_ind = [u, n_users + i]
#             details['knows_user'] = True
#             details['knows_item'] = True
#         elif self.trainset.knows_user(u):
#             data = [1.]
#             row_ind = [0]
#             col_ind = [u]
#             details['knows_user'] = True
#             details['knows_item'] = False
#         elif self.trainset.knows_item(i):
#             data = [1.]
#             row_ind = [0]
#             col_ind = [n_users + i]
#             details['knows_user'] = False
#             details['knows_item'] = True
#         else:
#             data = []
#             row_ind = []
#             col_ind = []
#             details['knows_user'] = False
#             details['knows_item'] = False

#         # Add user features (if any)
#         if n_user_features > 0:
#             for n, value in enumerate(u_features):
#                 if value != 0:
#                     row_ind.append(0)
#                     col_ind.append(n_users + n_items + n)
#                     data.append(value)

#         # Add item features (if any)
#         if n_item_features > 0:
#             for n, value in enumerate(i_features):
#                 if value != 0:
#                     row_ind.append(0)
#                     col_ind.append(n_users + n_items +
#                                    n_user_features + n)
#                     data.append(value)

#         if data:
#             X_test = sps.csr_matrix((data, (row_ind, col_ind)),
#                                     shape=(1, n_users + n_items +
#                                            n_user_features + n_item_features),
#                                     dtype=np.float32)
#         else:
#             X_test = sps.csr_matrix((1, n_users + n_items + n_user_features +
#                                      n_item_features),
#                                     dtype=np.float32)

#         est = self.model.predict(X_test)[0]

#         return est, details


# class FMBasicPL(AlgoBase):
#     """A basic factorization machine algorithm. With order 2, this algorithm
#     is equivalent to the biased SVD algorithm.

#     This code is an interface to the `polylearn` library.

#     Args:
#         degree : int, default: 2
#             Degree of the polynomial. Corresponds to the order of feature
#             interactions captured by the model. Currently only supports
#             degrees up to 3.
#         n_factors : int, default: 2
#             Number of basis vectors to learn, a.k.a. the dimension of the
#             low-rank parametrization.
#         reg_alpha : float, default: 1
#             Regularization amount for linear term (if ``fit_linear=True``).
#         reg_beta : float, default: 1
#             Regularization amount for higher-order weights.
#         tol : float, default: 1e-6
#             Tolerance for the stopping condition.
#         fit_lower : {'explicit'|'augment'|None}, default: 'explicit'
#             Whether and how to fit lower-order, non-homogeneous terms.
#             - 'explicit': fits a separate P directly for each lower order.
#             - 'augment': adds the required number of dummy columns (columns
#                that are 1 everywhere) in order to capture lower-order terms.
#                Adds ``degree - 2`` columns if ``fit_linear`` is true, or
#                ``degree - 1`` columns otherwise, to account for the linear
#                term.
#             - None: only learns weights for the degree given.  If
#               ``degree == 3``, for example, the model will only have weights
#               for third-order feature interactions.
#         fit_linear : {True|False}, default: True
#             Whether to fit an explicit linear term <w, x> to the model, using
#             coordinate descent. If False, the model can still capture linear
#             effects if ``fit_lower == 'augment'``.
#         warm_start : boolean, optional, default: False
#             Whether to use the existing solution, if available. Useful for
#             computing regularization paths or pre-initializing the model.
#         init_lambdas : {'ones'|'random_signs'}, default: 'ones'
#             How to initialize the predictive weights of each learned basis. The
#             lambdas are not trained; using alternate signs can theoretically
#             improve performance if the kernel degree is even.  The default
#             value of 'ones' matches the original formulation of factorization
#             machines (Rendle, 2010).
#             To use custom values for the lambdas, ``warm_start`` may be used.
#         max_iter : int, optional, default: 10000
#             Maximum number of passes over the dataset to perform.
#         random_state : int seed, RandomState instance, or None (default)
#             The seed of the pseudo random number generator to use for
#             initializing the parameters.
#         verbose : boolean, optional, default: False
#             Whether to print debugging information.
#     """

#     def __init__(self, degree=2, n_factors=2, reg_alpha=1., reg_beta=1.,
#                  tol=1e-6, fit_lower='explicit', fit_linear=True,
#                  warm_start=False, init_lambdas='ones', max_iter=10000,
#                  random_state=None, verbose=False, **kwargs):

#         self.degree = degree
#         self.n_factors = n_factors  # n_components in `polylearn`
#         self.reg_alpha = reg_alpha  # alpha in `polylearn`
#         self.reg_beta = reg_beta  # beta in `polylearn`
#         self.tol = tol
#         self.fit_lower = fit_lower
#         self.fit_linear = fit_linear
#         self.warm_start = warm_start
#         self.init_lambdas = init_lambdas
#         self.max_iter = max_iter
#         self.random_state = random_state
#         self.verbose = verbose

#         self.model = FactorizationMachineRegressor(
#             degree=self.degree, n_components=self.n_factors,
#             alpha=self.reg_alpha, beta=self.reg_beta, tol=self.tol,
#             fit_lower=self.fit_lower, fit_linear=self.fit_linear,
#             warm_start=self.warm_start, init_lambdas=self.init_lambdas,
#             max_iter=self.max_iter, random_state=self.random_state,
#             verbose=self.verbose)

#     def fit(self, trainset):

#         AlgoBase.fit(self, trainset)

#         n_ratings = self.trainset.n_ratings
#         n_users = self.trainset.n_users
#         n_items = self.trainset.n_items

#         # Construct sparse X and y
#         row_ind = np.empty(2 * n_ratings, dtype=int)
#         col_ind = np.empty(2 * n_ratings, dtype=int)
#         data = np.ones(2 * n_ratings, dtype=bool)
#         y_train = np.empty(n_ratings, dtype=float)
#         nonzero_counter = 0
#         rating_counter = 0
#         for uid, iid, rating in self.trainset.all_ratings():
#             # Add user
#             row_ind[nonzero_counter] = rating_counter
#             col_ind[nonzero_counter] = uid
#             nonzero_counter += 1
#             # Add item
#             row_ind[nonzero_counter] = rating_counter
#             col_ind[nonzero_counter] = n_users + iid
#             nonzero_counter += 1
#             # Add rating
#             y_train[rating_counter] = rating
#             rating_counter += 1
#         X_train = sps.csc_matrix((data, (row_ind, col_ind)),
#                                  shape=(n_ratings, n_users + n_items),
#                                  dtype=bool)

#         self.model.fit(X_train, y_train)
#         self.X_train = X_train
#         self.y_train = y_train

#         return self

#     def estimate(self, u, i, *_):

#         n_users = self.trainset.n_users
#         n_items = self.trainset.n_items

#         details = {}
#         if self.trainset.knows_user(u) and self.trainset.knows_item(i):
#             X_test = sps.csc_matrix(([1., 1.], ([0, 0], [u, n_users + i])),
#                                     shape=(1, n_users + n_items), dtype=bool)
#             details['knows_user'] = True
#             details['knows_item'] = True
#         elif self.trainset.knows_user(u):
#             X_test = sps.csc_matrix(([1.], ([0], [u])),
#                                     shape=(1, n_users + n_items), dtype=bool)
#             details['knows_user'] = True
#             details['knows_item'] = False
#         elif self.trainset.knows_item(i):
#             X_test = sps.csc_matrix(([1.], ([0], [n_users + i])),
#                                     shape=(1, n_users + n_items), dtype=bool)
#             details['knows_user'] = False
#             details['knows_item'] = True
#         else:
#             X_test = sps.csc_matrix((1, n_users + n_items), dtype=bool)
#             details['knows_user'] = False
#             details['knows_item'] = False

#         est = self.model.predict(X_test)[0]

#         return est, details
