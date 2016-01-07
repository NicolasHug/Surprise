from collections import defaultdict
import pickle
import time
import os
import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})

import similarities as sims
import colors


class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible"""
    pass


class AlgoBase:
    """Abstract Algo class where is defined the basic behaviour of a recomender
    algorithm"""

    def __init__(self, training_data, item_based=False, with_dump=False, **kwargs):

        self.training_data = training_data

        # whether the algo will be based on users (basically means that the
        # similarities will be computed between users or between items)
        # if the algo is user based, x denotes a user and y an item
        # if the algo is item based, x denotes an item and y a user
        self.ub = not item_based

        if self.ub:
            self.rm = training_data.rm
            self.xr = training_data.ur
            self.yr = training_data.ir
            self.n_x = training_data.n_users
            self.n_y = training_data.n_items
        else:
            self.rm = defaultdict(int)
            # @TODO: maybe change that...
            for (ui, mi), r in training_data.rm.items():
                self.rm[mi, ui] = r
            self.xr = training_data.ir
            self.yr = training_data.ur
            self.n_x = training_data.n_items
            self.n_y = training_data.n_users

        # global mean of all ratings
        self.global_mean = np.mean([r for (_, _, r) in self.all_ratings])
        # list of all predictions computed by the algorithm
        self.preds = []

        self.with_dump = with_dump
        self.infos = {}
        self.infos['name'] = 'undefined'
        self.infos['params'] = {}  # dict of params specific to any algo
        self.infos['params']['Based on '] = 'users' if self.ub else 'items'
        self.infos['ub'] = self.ub
        self.infos['preds'] = self.preds  # list of predictions.
        self.infos['ur'] = training_data.ur  # user ratings  dict
        self.infos['ir'] = training_data.ir  # item ratings dict
        self.infos['rm'] = self.rm  # rating matrix
        # Note: there is a lot of duplicated data, the dumped file will be
        # HUGE.

    def predict(self, u0, i0, r0=0, output=False):
        """Predict the rating for u0 and i0 by calling the estimate method of
        the algorithm (defined in every sub-class). If prediction is impossible
        (for any reason), set prediction to the global mean of all ratings. the
        self.preds attribute is updated.
        """

        try:
            est = self.estimate(u0, i0)
            impossible = False
        except PredictionImpossible:
            est = self.global_mean
            impossible = True

        # clip estimate into [self.r_min, self.r_max]
        est = min(self.training_data.r_max, est)
        est = max(self.training_data.r_min, est)

        if output:
            if impossible:
                print(colors.FAIL + 'Impossible to predict' + colors.ENDC)
            err = abs(est - r0)
            col = colors.FAIL if err > 1 else colors.OKGREEN
            print(col + "err = {0:1.2f}".format(err) + colors.ENDC)

        pred = (u0, i0, r0, est, impossible)
        self.preds.append(pred)

        return pred

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

    def dump_infos(self):
        """dump the dict self.infos into a file"""

        if not self.with_dump:
            return
        if not os.path.exists('./dumps'):
            os.makedirs('./dumps')

        date = time.strftime('%y%m%d-%Hh%Mm%S', time.localtime())
        name = ('dumps/' + date + '-' + self.infos['name'] + '-' +
                str(len(self.infos['preds'])))
        pickle.dump(self.infos, open(name, 'wb'))

    def getx0y0(self, u0, i0):
        """return x0 and y0 based on the self.ub variable (see constructor)"""
        if self.ub:
            return u0, i0
        else:
            return i0, u0


class AlgoUsingSim(AlgoBase):
    """Abstract class for algos using a similarity measure"""

    def __init__(self, training_data, item_based, sim_name, **kwargs):
        super().__init__(training_data, item_based, **kwargs)

        self.infos['params']['sim'] = sim_name
        self.construct_sim_mat(sim_name)  # we'll need the similiarities

    def construct_sim_mat(self, sim_name):
        """construct the simlarity matrix"""

        print("computing the similarity matrix...")
        construction_func = {'cos' : sims.cosine,
                             'MSD' : sims.msd,
                             'MSDClone' : sims.msdClone,
                             'pearson' : sims.pearson}

        try:
            self.sim = construction_func[sim_name](self.n_x, self.yr)
        except KeyError:
            raise NameError('Wrong sim name')

class AlgoWithBaseline(AlgoBase):
    """Abstract class for algos that need a baseline"""

    def __init__(self, training_data, item_based, method, **kwargs):
        super().__init__(training_data, item_based, **kwargs)

        # compute users and items biases
        # see from 5.2.1 of RS handbook

        self.x_biases = np.zeros(self.n_x)
        self.y_biases = np.zeros(self.n_y)

        print('Estimating biases...')
        if method == 'sgd':
            self.optimize_sgd()
        elif method == 'als':
            self.optimize_als()

    def optimize_sgd(self):
        """optimize biases using sgd"""
        lambda4 = 0.02
        gamma = 0.005
        n_epochs = 20
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

    def optimize_als(self):
        """alternatively optimize y_biases and x_biases. Probably not really an
        als"""
        reg_u = 15
        reg_i = 10
        n_epochs = 10

        self.reg_x = reg_u if self.ub else reg_i
        self.reg_y = reg_u if not self.ub else reg_i

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

    def get_baseline(self, x, y):
        return self.global_mean + self.x_biases[x] + self.y_biases[y]
