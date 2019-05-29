"""
the :mod:`knns` module includes some k-NN inspired algorithms.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from .predictions import PredictionImpossible
from .algo_base import AlgoBase


class FMtorchNN(nn.Module):
    """ The PyTorch model for factorization machine. This class is used by
    `FM`. The initilization is done as in Rendle (2012).

    Args:
        n_features: int
            Defines the number of features in x.
        n_factors: int, default: 20
            Defines the number of factors in the interaction terms.
        init_std: float, default: 0.01
            The standard deviation of the normal distribution for
            initialization.
    """

    def __init__(self, n_features, n_factors=20, init_std=0.01):
        super(FMtorchNN, self).__init__()
        self.n_features = n_features
        self.n_factors = n_factors
        self.init_std = init_std

        # Initialize bias term
        self.b = nn.Parameter(torch.Tensor(1),
                              requires_grad=True)
        self.b.data.fill_(0.)
        # self.b.data.normal_(init_mean, init_std)
        # self.b.data.uniform_(-0.01, 0.01)

        # Initialize linear terms
        self.w = nn.Parameter(torch.Tensor(self.n_features, 1),
                              requires_grad=True)
        self.w.data.fill_(0.)
        # self.w.data.normal_(init_mean, init_std)
        # self.w.data.uniform_(-0.01, 0.01)

        # Initialize interaction terms
        self.V = nn.Parameter(torch.Tensor(self.n_features, self.n_factors),
                              requires_grad=True)
        self.V.data.normal_(0., self.init_std)
        # self.V.data.uniform_(-0.01, 0.01)

    def forward(self, x):

        # The linear part
        total_linear = torch.sum(torch.mm(x, self.w), dim=1)

        # The interaction part
        # O(kn) formulation from Steffen Rendle
        total_inter_1 = torch.mm(x, self.V) ** 2
        total_inter_2 = torch.mm(x ** 2, self.V ** 2)
        total_inter = 0.5 * torch.sum(total_inter_1 - total_inter_2, dim=1)

        # Compute predictions
        y_pred = self.b + total_linear + total_inter

        return y_pred


class FM(AlgoBase):
    """A factorization machine algorithm implemented using pytorch.

    Args:
        rating_lst : list of str or `None`, default : ['userID', 'itemID']
            This list specifies what information from the `raw_ratings` to put
            in the `x` vector. Accepted list values are 'userID', 'itemID',
            'imp_u_rating' and 'exp_u_rating'. Implicit and explicit user
            rating values are scaled by the number of values. If `None`, no
            info is added.
        user_lst : list of str or `None`, default : `None`
            This list specifies what information from the `user_features` to
            put in the `x` vector. Accepted list values consist of the names of
            features. If `None`, no info is added.
        item_lst : list of str or `None`, default : `None`
            This list specifies what information from the `item_features` to
            put in the `x` vector. Accepted list values consist of the names of
            features. If `None`, no info is added.
        n_factors : int, default: 20
            Number of latent factors in low-rank appoximation.
        n_epochs : int, default : 30
            Number of epochs. All epochs are ran but only the best model out of
            all epochs is kept.
        dev_ratio : float, default : 0.3
            Ratio of `trainset` to dedicate to development data set to identify
            best model. Should be either positive and smaller than the number
            of samples or a float in the (0, 1) range.
        init_std: float, default : 0.01
            The standard deviation of the normal distribution for
            initialization.
        lr : float, default: 0.001
            Learning rate for optimization method.
        reg : float, default: 0.02
            Strength of L2 regularization. It can be disabled by setting it to
            zero.
        random_state : int, default: `None`
            Determines the RNG that will be used for initialization. If
            int, ``random_state`` will be used as a seed for a new RNG. This is
            useful to get the same initialization over multiple calls to
            ``fit()``. If ``None``, the current RNG from torch is used.
        verbose : int, default: False
            Level of verbosity.
    """

    def __init__(self, rating_lst=['userID', 'itemID'], user_lst=None,
                 item_lst=None, n_factors=20, n_epochs=30, dev_ratio=0.3,
                 init_std=0.01, lr=0.001, reg=0.02, random_state=None,
                 verbose=False, **kwargs):

        AlgoBase.__init__(self, **kwargs)
        self.rating_lst = rating_lst
        self.user_lst = user_lst
        self.item_lst = item_lst
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.dev_ratio = dev_ratio
        self.init_std = init_std
        self.lr = lr
        self.reg = reg
        self.random_state = random_state
        self.verbose = verbose

        torch.set_default_dtype(torch.float64)  # use float64

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        # Construct data and initialize model
        # Initialization needs to be done in fit() since it depends on the
        # trainset
        if self.random_state:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
        self._construct_FM_data()
        self.model = FMtorchNN(self.n_features, self.n_factors, self.init_std)
        params = FM._add_weight_decay(self.model, self.reg)
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        # Define data (TODO : sample_weights)
        x = self.libsvm_df.loc[:, self.libsvm_df.columns != 'rating'].values
        y = self.libsvm_df.loc[:, 'rating'].values
        sample_weights = None
        if sample_weights:
            x_train, x_dev, y_train, y_dev, w_train, w_dev = train_test_split(
                x, y, sample_weights, test_size=self.dev_ratio,
                random_state=self.random_state)
            w_train = torch.Tensor(w_train)
            w_dev = torch.Tensor(w_dev)
        else:
            x_train, x_dev, y_train, y_dev = train_test_split(
                x, y, test_size=self.dev_ratio, random_state=self.random_state)
            w_train = None
            w_dev = None
        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)
        x_dev = torch.Tensor(x_dev)
        y_dev = torch.Tensor(y_dev)

        best_loss = np.inf
        best_model = None
        for epoch in range(self.n_epochs):
            # Switch to training mode, clear gradient accumulators
            self.model.train()
            self.optimizer.zero_grad()
            # Forward pass
            y_pred = self.model(x_train)
            # Compute loss
            self.train_loss = self._compute_loss(y_pred, y_train, w_train)
            # Backward pass and update weights
            self.train_loss.backward()
            self.optimizer.step()

            # Switch to eval mode and evaluate with development data
            # See https://github.com/pytorch/examples/blob/master/snli/train.py
            self.model.eval()
            y_pred = self.model(x_dev)
            self.dev_loss = self._compute_loss(y_pred, y_dev, w_dev)

            if self.verbose:
                print(epoch, self.train_loss.item(), self.dev_loss.item())

            if self.dev_loss.item() < best_loss:
                best_model = copy.deepcopy(self.model)
                best_loss = self.dev_loss.item()
                if self.verbose:
                    print('A new best model have been found!')

        self.model = best_model

        return self

    def _add_weight_decay(model, reg, skip_list=[]):
        """ Add weight_decay with no regularization for bias.
        """

        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if ((len(param.shape) == 1) or name.endswith(".bias") or
                    (name in skip_list)):
                no_decay.append(param)
            else:
                decay.append(param)

        return [{'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': reg}]

    def _compute_loss(self, y_pred, y, sample_weights=None):
        """ Computes a different loss depending on whether `sample_weights` are
        defined.
        """

        if sample_weights is not None:
            criterion = nn.MSELoss(reduction='none')
            loss = criterion(y_pred, y)
            loss = torch.dot(sample_weights, loss) / y.shape[0]
        else:
            criterion = nn.MSELoss()
            loss = criterion(y_pred, y)

        return loss

    def _construct_FM_data(self):
        """ Construct the data needed by `FM`.

        It is assumed that the user and item features are correctly encoded.
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
        self.n_features = libsvm_df.shape[1] - 1

    def estimate(self, u, i, u_features, i_features):

        # Estimate rating
        x = self._construct_estimate_input(u, i, u_features, i_features)
        x = torch.Tensor(x[None, :])  # add dimension
        est = float(self.model(x))

        # Construct details
        details = {}
        if self.trainset.knows_user(u) and self.trainset.knows_item(i):
            details['knows_user'] = True
            details['knows_item'] = True
        elif self.trainset.knows_user(u):
            details['knows_user'] = True
            details['knows_item'] = False
        elif self.trainset.knows_item(i):
            details['knows_user'] = False
            details['knows_item'] = True
        else:
            details['knows_user'] = False
            details['knows_item'] = False

        return est, details

    def _construct_estimate_input(self, u, i, u_features, i_features):
        """ Construct the input for the model.

        It is assumed that if features are given in u_features or i_features,
        they are all given and in the same order as in the trainset.
        """

        n_users = self.trainset.n_users
        n_items = self.trainset.n_items

        x = []

        # Add rating features
        if self.rating_lst:
            for feature in self.rating_lst:
                if feature == 'userID':
                    temp = [0.] * n_users
                    if self.trainset.knows_user(u):
                        temp[u] = 1.
                    x.extend(temp)
                elif feature == 'itemID':
                    temp = [0.] * n_items
                    if self.trainset.knows_item(i):
                        temp[i] = 1.
                    x.extend(temp)
                elif feature == 'imp_u_rating':
                    temp = [0.] * n_items
                    if self.trainset.knows_user(u):
                        all_u_ratings = self.trainset.ur[u]
                        for other_i, rating in all_u_ratings:
                            if other_i != i:  # only the other ratings
                                temp[other_i] = 1.
                        temp = np.array(temp)
                        count = np.count_nonzero(temp)
                        if count == 0:
                            count = 1
                        temp = list(temp / count)
                    x.extend(temp)
                elif feature == 'exp_u_rating':
                    # a rating is at least 1 with the offset
                    temp = [0.] * n_items
                    if self.trainset.knows_user(u):
                        all_u_ratings = self.trainset.ur[u]
                        for other_i, rating in all_u_ratings:
                            if other_i != i:  # only the other ratings
                                temp[other_i] = rating
                        temp = np.array(temp)
                        count = np.count_nonzero(temp)
                        if count == 0:
                            count = 1
                        temp = list(temp / count)
                    x.extend(temp)

        # Add user features
        if self.user_lst:
            temp = [0.] * len(self.user_lst)
            if u_features:
                # It is assumed that if features are given, they are all given.
                temp_df = pd.Series(
                    u_features, index=self.trainset.user_features_labels)
                for idx, feature in enumerate(self.user_lst):
                    temp[idx] = temp_df[feature]
            x.extend(temp)

        # Add item features
        if self.item_lst:
            temp = [0.] * len(self.item_lst)
            if u_features:
                # It is assumed that if features are given, they are all given.
                temp_df = pd.Series(
                    i_features, index=self.trainset.item_features_labels)
                for idx, feature in enumerate(self.item_lst):
                    temp[idx] = temp_df[feature]
            x.extend(temp)

        return np.array(x)
