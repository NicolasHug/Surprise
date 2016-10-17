from collections import defaultdict
from collections import namedtuple
import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})

from .. import similarities as sims
from .. import colors


class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible"""
    pass

Prediction = namedtuple('Prediction', ['uid', 'iid', 'r0', 'est', 'details'])

class AlgoBase:
    """Abstract Algo class where is defined the basic behaviour of a
    recommender algorithm"""

    def __init__(self, user_based=True, **kwargs):

        # whether the algo will be based on users (basically means that the
        # similarities will be computed between users or between items)
        # if the algo is user based, x denotes a user and y an item
        # if the algo is item based, x denotes an item and y a user
        self.user_based = user_based

        self.bsl_options = kwargs.get('baseline', {})
        self.sim_options = kwargs.get('sim', {})

    def train(self, trainset):
        self.trainset = trainset

        if self.user_based:
            self.rm = trainset.rm
            self.xr = trainset.ur
            self.yr = trainset.ir
            self.n_x = trainset.n_users
            self.n_y = trainset.n_items
        else:
            self.rm = defaultdict(int)
            # @TODO: maybe change that...
            for (ui, mi), r in trainset.rm.items():
                self.rm[mi, ui] = r
            self.xr = trainset.ir
            self.yr = trainset.ur
            self.n_x = trainset.n_items
            self.n_y = trainset.n_users

        # number of ratings
        self.n_ratings = len(self.rm)
        # global mean of all ratings
        self.global_mean = np.mean([r for (_, _, r) in self.all_ratings])

        self.x_biases = self.y_biases = None
        self.sim = None

    def predict(self, u0, i0, r0=0, output=False):
        """Predict the rating for u0 and i0 by calling the estimate method of
        the algorithm (defined in every sub-class). If prediction is impossible
        (for any reason), set estimation to the global mean of all ratings.
        """

        x0, y0 = (u0, i0) if self.user_based else (i0, u0)

        # TODO: handle prediction details in a better way (possibly avoiding
        # side effects)
        self.pred_details= {}

        try:
            if u0 == 'unknown' or i0 == 'unknown':
                raise PredictionImpossible('user or item was not part of ' +
                                           'training set')

            est = self.estimate(x0, y0)
            impossible = False
        except PredictionImpossible:
            est = self.global_mean
            impossible = True

        # clip estimate into [self.r_min, self.r_max]
        est = min(self.trainset.r_max, est)
        est = max(self.trainset.r_min, est)

        if output:
            if impossible:
                print(colors.FAIL + 'Impossible to predict' + colors.ENDC)
            err = abs(est - r0)
            col = colors.FAIL if err > 1 else colors.OKGREEN
            print(col + "err = {0:1.2f}".format(err) + colors.ENDC)

        self.pred_details['was_impossible'] = impossible
        return Prediction(u0, i0, r0, est, self.pred_details)

    def test(self, testset):

        predictions = [self.predict(uid, iid, r) for (uid, iid, r) in testset]
        return predictions

    def compute_baselines(self):
        """Compute users and items biases. See from 5.2.1 of RS handbook"""

        # if this method has already been called before on the same trainset,
        # then don't do anything.  it's useful to handle cases where the
        # similarity metric (pearson_baseline for example) uses baseline
        # estimates.
        # I don't quite like the way it's handled but it works...
        if self.x_biases is not None:
            return

        def optimize_sgd():
            """optimize biases using sgd"""

            lambda4 = self.bsl_options.get('lambda4', 0.02)
            gamma = self.bsl_options.get('gamma', 0.005)
            n_epochs = self.bsl_options.get('n_epochs', 20)

            for dummy in range(n_epochs):
                for x, y, r in self.all_ratings:
                    err = (r -
                          (self.global_mean + self.x_biases[x] + self.y_biases[y]))
                    # update x_biases
                    self.x_biases[x] += gamma * (err - lambda4 *
                                                self.x_biases[x])
                    # udapte y_biases
                    self.y_biases[y] += gamma * (err - lambda4 *
                                                self.y_biases[y])

        def optimize_als():
            """alternatively optimize y_biases and x_biases. Probably not
            really an als"""

            reg_u = self.bsl_options.get('reg_u', 15)
            reg_i = self.bsl_options.get('reg_i', 10)
            n_epochs = self.bsl_options.get('n_epochs', 10)

            self.reg_x = reg_u if self.user_based else reg_i
            self.reg_y = reg_i if self.user_based else reg_u

            for dummy in range(n_epochs):
                self.y_biases = np.zeros(self.n_y)
                for y in self.all_ys:
                    devY = sum(r - self.global_mean -
                               self.x_biases[x] for (x, r) in self.yr[y])
                    self.y_biases[y] = devY / (self.reg_y + len(self.yr[y]))

                self.x_biases = np.zeros(self.n_x)
                for x in self.all_xs:
                    devX = sum(r - self.global_mean -
                               self.y_biases[y] for (y, r) in self.xr[x])
                    self.x_biases[x] = devX / (self.reg_x + len(self.xr[x]))

        self.x_biases = np.zeros(self.n_x)
        self.y_biases = np.zeros(self.n_y)

        optimize = dict(als=optimize_als,
                        sgd=optimize_sgd)

        method = self.bsl_options.get('method', 'als')

        try:
            print('Estimating biases...')
            optimize[method]()
        except KeyError:
            raise ValueError('invalid method ' + method + ' for baseline ' +
                             'computation. Available methods are als, sgd.')


    def get_baseline(self, x, y):
        return self.global_mean + self.x_biases[x] + self.y_biases[y]

    def compute_similarities(self):
        """construct the simlarity matrix"""

        print("computing the similarity matrix...")
        construction_func = {'cos' : sims.cosine,
                             'MSD' : sims.msd,
                             'pearson' : sims.pearson,
                             'pearson_baseline' : sims.pearson_baseline}

        name = self.sim_options.get('name', 'MSD')
        args = [self.n_x, self.yr]
        if name == 'pearson_baseline':
            shrinkage = self.sim_options.get('shrinkage', 100)
            self.compute_baselines()
            args += [self.global_mean, self.x_biases, self.y_biases, shrinkage]

        try:
            self.sim = construction_func[name](*args)
        except KeyError:
            raise NameError('Wrong sim name ' + name + '. Allowed values ' +
                            'are ' + ', '.join(construction_func.keys()) + '.')

    @property
    def all_ratings(self):
        """generator to iter over all ratings"""

        for x, x_ratings in self.xr.items():
            for y, r in x_ratings:
                yield x, y, r

    @property
    def all_xs(self):
        """generator to iter over all xs"""
        return range(self.n_x)

    @property
    def all_ys(self):
        """generator to iter over all ys"""
        return range(self.n_y)
