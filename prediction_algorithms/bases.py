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

    def __init__(self, trainingData, itemBased=False, withDump=False, **kwargs):

        self.trainingData = trainingData

        # whether the algo will be based on users (basically means that the
        # similarities will be computed between users or between items)
        # if the algo is user based, x denotes a user and y an item
        # if the algo is item based, x denotes an item and y a user
        self.ub = not itemBased

        if self.ub:
            self.rm = trainingData.rm
            self.xr = trainingData.ur
            self.yr = trainingData.ir
            self.nX = trainingData.nUsers
            self.nY = trainingData.nItems
        else:
            self.rm = defaultdict(int)
            # @TODO: maybe change that...
            for (ui, mi), r in trainingData.rm.items():
                self.rm[mi, ui] = r
            self.xr = trainingData.ir
            self.yr = trainingData.ur
            self.nX = trainingData.nItems
            self.nY = trainingData.nUsers

        # global mean of all ratings
        self.meanRatings = np.mean([r for (_, _, r) in self.allRatings])
        # list of all predictions computed by the algorithm
        self.preds = []

        self.withDump = withDump
        self.infos = {}
        self.infos['name'] = 'undefined'
        self.infos['params'] = {}  # dict of params specific to any algo
        self.infos['params']['Based on '] = 'users' if self.ub else 'items'
        self.infos['ub'] = self.ub
        self.infos['preds'] = self.preds  # list of predictions.
        self.infos['ur'] = trainingData.ur  # user ratings  dict
        self.infos['ir'] = trainingData.ir  # item ratings dict
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
            est = self.meanRatings
            impossible = True

        # clip estimate into [self.rMin, self.rMax]
        est = min(self.trainingData.rMax, est)
        est = max(self.trainingData.rMin, est)

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
    def allRatings(self):
        """generator to iter over all ratings"""

        for x, xRatings in self.xr.items():
            for y, r in xRatings:
                yield x, y, r

    @property
    def allXs(self):
        """generator to iter over all xs"""
        return range(self.nX)

    @property
    def allYs(self):
        """generator to iter over all ys"""
        return range(self.nY)

    def dumpInfos(self):
        """dump the dict self.infos into a file"""

        if not self.withDump:
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

    def __init__(self, trainingData, itemBased, sim, **kwargs):
        super().__init__(trainingData, itemBased, **kwargs)

        self.infos['params']['sim'] = sim
        self.constructSimMat(sim)  # we'll need the similiarities

    def constructSimMat(self, sim):
        """construct the simlarity matrix"""

        print("computing the similarity matrix...")
        sim_measure = {'cos' : sims.cosine,
                       'MSD' : sims.msd,
                       'MSDClone' : sims.msdClone,
                       'pearson' : sims.pearson}

        try:
            self.simMat = sim_measure[sim](self.nX, self.yr)
        except KeyError:
            raise NameError('Wrong sim name')

class AlgoWithBaseline(AlgoBase):
    """Abstract class for algos that need a baseline"""

    def __init__(self, trainingData, itemBased, method, **kwargs):
        super().__init__(trainingData, itemBased, **kwargs)

        # compute users and items biases
        # see from 5.2.1 of RS handbook

        self.xBiases = np.zeros(self.nX)
        self.yBiases = np.zeros(self.nY)

        print('Estimating biases...')
        if method == 'sgd':
            self.optimize_sgd()
        elif method == 'als':
            self.optimize_als()

    def optimize_sgd(self):
        """optimize biases using sgd"""
        lambda4 = 0.02
        gamma = 0.005
        nIter = 20
        for dummy in range(nIter):
            for x, y, r in self.allRatings:
                err = (r -
                      (self.meanRatings + self.xBiases[x] + self.yBiases[y]))
                # update xBiases
                self.xBiases[x] += gamma * (err - lambda4 *
                                            self.xBiases[x])
                # udapte yBiases
                self.yBiases[y] += gamma * (err - lambda4 *
                                            self.yBiases[y])

    def optimize_als(self):
        """alternatively optimize yBiases and xBiases. Probably not really an
        als"""
        reg_u = 15
        reg_i = 10
        nIter = 10

        self.reg_x = reg_u if self.ub else reg_i
        self.reg_y = reg_u if not self.ub else reg_i

        for dummy in range(nIter):
            self.yBiases = np.zeros(self.nY)
            for y in self.allYs:
                devY = sum(r - self.meanRatings -
                           self.xBiases[x] for (x, r) in self.yr[y])
                self.yBiases[y] = devY / (self.reg_y + len(self.yr[y]))

            self.xBiases = np.zeros(self.nX)
            for x in self.allXs:
                devX = sum(r - self.meanRatings -
                           self.yBiases[y] for (y, r) in self.xr[x])
                self.xBiases[x] = devX / (self.reg_x + len(self.xr[x]))

    def getBaseline(self, x, y):
        return self.meanRatings + self.xBiases[x] + self.yBiases[y]
