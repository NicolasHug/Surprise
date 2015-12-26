from itertools import combinations
from collections import defaultdict
from scipy.stats import rv_discrete
import numpy as np
import pickle
import time
import os

import common as cmn

class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible"""
    pass

class Algo:
    """Abstract Algo class where is defined the basic behaviour of a recomender
    algorithm"""
    def __init__(self, trainingData, itemBased=False, withDump=False):

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
        self.meanRatings = np.mean([r for (_, _, r) in self.iterAllRatings()])
        # list of all predictions computed by the algorithm
        self.preds = []

        self.withDump = withDump
        self.infos = {}
        self.infos['name'] = 'undefined'
        self.infos['params'] = {} # dict of params specific to any algo
        self.infos['params']['Based on '] = 'users' if self.ub else 'items'
        self.infos['ub'] = self.ub
        self.infos['preds'] = self.preds  # list of predictions.
        self.infos['ur'] = trainingData.ur # user ratings  dict
        self.infos['ir'] = trainingData.ir # item ratings dict
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
                print(cmn.Col.FAIL + 'Impossible to predict' + cmn.Col.ENDC)
            err = abs(est - r0)
            col = cmn.Col.FAIL if err > 1 else cmn.Col.OKGREEN
            print(col + "err = {0:1.2f}".format(err) + cmn.Col.ENDC)

        pred = (u0, i0, r0, est, impossible)
        self.preds.append(pred)

        return pred

    def iterAllRatings(self):
        """generator to iter over all ratings"""

        for x, xRatings in self.xr.items():
            for y, r in xRatings:
                yield x, y, r

    def dumpInfos(self):
        """dump the dict self.infos into a file"""

        if not self.withDump:
            return
        if not os.path.exists('./dumps'):
            os.makedirs('./dumps')

        date = time.strftime('%y%m%d-%Hh%Mm%S', time.localtime())
        name = ('dumps/' + date + '-' + self.infos['name'] + '-' +
            str(len(self.infos['preds'])))
        pickle.dump(self.infos, open(name,'wb'))

    def getx0y0(self, u0, i0):
        """return x0 and y0 based on the self.ub variable (see constructor)"""
        if self.ub:
            return u0, i0
        else:
            return i0, u0

class AlgoRandom(Algo):
    """predict a random rating based on the distribution of the training set"""

    def __init__(self, trainingData, **kwargs):
        super().__init__(trainingData)
        self.infos['name'] = 'random'

        # estimation of the distribution
        fqs = [0, 0, 0, 0, 0]
        for x in range(nX):
            for y in range(nY):
                if self.rm[x, y] > 0:
                    fqs[self.rm[x, y] - 1] += 1
        fqs = [fq/sum(fqs) for fq in fqs]
        self.distrib = rv_discrete(values=([1, 2, 3, 4, 5], fqs))

    def estimate(self, *_):
        return self.distrib.rvs()

class AlgoUsingSim(Algo):
    """Abstract class for algos using a similarity measure
    sim parameter can be 'cos' or 'MSD' for mean squared difference"""
    def __init__(self, trainingData, itemBased, sim, **kwargs):
        super().__init__(trainingData, itemBased, **kwargs)

        self.infos['params']['sim'] = sim
        self.constructSimMat(sim) # we'll need the similiarities

    def constructSimMat(self, sim):
        """construct the simlarity matrix"""

        print("computing the similarity matrix...")
        self.simMat = np.zeros((self.nX, self.nX))
        if sim == 'cos':
            self.constructCosineSimMat()
        elif sim == 'MSD':
            self.constructMSDSimMat()
        elif sim == 'MSDClone':
            self.constructMSDCloneSimMat()
        elif sim == 'pearson':
            self.constructPearsonSimMat()
        else:
            raise NameError('WrongSimName')

    def constructCosineSimMat(self):
        """compute the cosine similarity between all pairs of xs.

        Technique inspired from MyMediaLite"""

        prods = defaultdict(int)  # sum (r_ui * r_vi) for common items
        freq = defaultdict(int)   # number common items
        sqi = defaultdict(int)  # sum (r_ui ^ 2) for common items
        sqj = defaultdict(int)  # sum (r_vi ^ 2) for common items

        for y, yRatings in self.yr.items():
            for (xi, r1), (xj, r2) in combinations(yRatings, 2):
                # note : accessing and updating elements takes a looooot of
                # time. Yet defaultdict is still faster than a numpy array...
                prods[xi, xj] += r1 * r2
                freq[xi, xj] += 1
                sqi[xi, xj] += r1**2
                sqj[xi, xj] += r2**2

        for xi in range(self.nX):
            self.simMat[xi, xi] = 1
            for xj in range(xi + 1, self.nX):
                if freq[xi, xj] == 0:
                    self.simMat[xi, xj] = 0
                else:
                    denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                    self.simMat[xi, xj] = prods[xi, xj] / denum

                self.simMat[xj, xi] = self.simMat[xi, xj]

    def constructMSDSimMat(self):
        """compute the mean squared difference similarity between all pairs of
        xs. MSDSim(xi, xj) = 1/MSD(xi, xj). if MSD(xi, xj) == 0, then
        MSDSim(xi, xj) = number of common ys. Implicitely, if there are no
        common ys, sim will be zero

        Technique inspired from MyMediaLite"""

        sqDiff = defaultdict(int)  # sum (r_ui - r_vi)**2 for common items
        freq = defaultdict(int)   # number common items

        for y, yRatings in self.yr.items():
            for (xi, r1), (xj, r2) in combinations(yRatings, 2):
                # note : accessing and updating elements takes a looooot of
                # time. Yet defaultdict is still faster than a numpya array...
                sqDiff[xi, xj] += (r1 - r2)**2
                freq[xi, xj] += 1

        for xi in range(self.nX):
            self.simMat[xi, xi] = 100 # completely arbitrary and useless anyway
            for xj in range(xi, self.nX):
                if sqDiff[xi, xj] == 0:  # return number of common ys
                    self.simMat[xi, xj] = freq[xi, xj]
                else:  # return inverse of MSD
                    self.simMat[xi, xj] = freq[xi, xj] / sqDiff[xi, xj]

                self.simMat[xj, xi] = self.simMat[xi, xj]

    def constructMSDCloneSimMat(self):
        """compute the 'clone' mean squared difference similarity between all
        pairs of xs. Some properties as for MSDSim apply"""

        for xi in range(self.nX):
            self.simMat[xi, xi] = 100 # completely arbitrary and useless anyway
            for xj in range(xi, self.nX):
                # comon ys for xi and xj
                Yij = [y for (y, _) in self.xr[xi] if self.rm[xj, y] > 0]

                if not Yij:
                    self.simMat[xi, xj] = 0
                    continue

                meanDiff = np.mean([self.rm[xi, y] - self.rm[xj, y] for y in Yij])
                # sum of squared differences:
                ssd = sum((self.rm[xi, y] - self.rm[xj, y] - meanDiff)**2 for y in Yij)
                if ssd == 0:
                    self.simMat[xi, xj] = len(Yij) # well... ok.
                else:
                    self.simMat[xi, xj] = len(Yij) / ssd

    def constructPearsonSimMat(self):
        """compute the pearson corr coeff between all pairs of xs.

        Technique inspired from MyMediaLite"""

        freq = defaultdict(int)   # number common items
        prods = defaultdict(int)  # sum (r_ui * r_vi) for common items
        sqi = defaultdict(int)  # sum (r_ui ^ 2) for common items
        sqj = defaultdict(int)  # sum (r_vi ^ 2) for common items
        si = defaultdict(int)  # sum (r_ui) for common items
        sj = defaultdict(int)  # sum (r_vi) for common items

        for y, yRatings in self.yr.items():
            for (xi, r1), (xj, r2) in combinations(yRatings, 2):
                # note : accessing and updating elements takes a looooot of
                # time. Yet defaultdict is still faster than a numpy array...
                prods[xi, xj] += r1 * r2
                freq[xi, xj] += 1
                sqi[xi, xj] += r1**2
                sqj[xi, xj] += r2**2
                si[xi, xj] += r1
                sj[xi, xj] += r2

        for xi in range(self.nX):
            self.simMat[xi, xi] = 1
            for xj in range(xi + 1, self.nX):
                n = freq[xi, xj]
                if n < 2:
                    self.simMat[xi, xj] = 0
                else:
                    num = n * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
                    denum = np.sqrt((n * sqi[xi, xj] - si[xi, xj]**2) *
                                    (n * sqj[xi, xj] - sj[xi, xj]**2))
                    if denum == 0:
                        self.simMat[xi, xj] = 0
                    else:
                        self.simMat[xi, xj] = num / denum

                self.simMat[xj, xi] = self.simMat[xi, xj]


class AlgoKNNBasic(AlgoUsingSim):
    """Basic collaborative filtering algorithm"""

    def __init__(self, trainingData, itemBased=False, sim='cos', k=40, **kwargs):
        super().__init__(trainingData, itemBased=itemBased, sim=sim)

        self.k = k

        self.infos['name'] = 'KNNBasic'
        self.infos['params']['similarity measure'] = sim
        self.infos['params']['k'] = self.k

    def estimate(self, u0, i0):
        x0, y0 = self.getx0y0(u0, i0)

        neighbors = [(x, self.simMat[x0, x], r) for (x, r) in self.yr[y0]]

        if not neighbors:
            raise PredictionImpossible

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x:x[1], reverse=True)

        # compute weighted average
        sumSim = sumRatings = 0
        for (nb, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sumSim += sim
                sumRatings += sim * r

        try:
            est = sumRatings / sumSim
        except ZeroDivisionError:
            raise PredictionImpossible

        return est

class AlgoKNNWithMeans(AlgoUsingSim):
    """Basic collaborative filtering algorithm, taking into account the mean
    ratings of each user"""

    def __init__(self, trainingData, itemBased=False, sim='cos', k=40, **kwargs):
        super().__init__(trainingData, itemBased=itemBased, sim=sim)

        self.k = k

        self.infos['name'] = 'basicWithMeans'
        self.infos['params']['similarity measure'] = sim
        self.infos['params']['k'] = self.k

        self.means = np.zeros(self.nX)
        for x, ratings in self.xr.items():
            self.means[x] = np.mean([r for (_, r) in self.xr[x]])

    def estimate(self, u0, i0):
        x0, y0 = self.getx0y0(u0, i0)

        neighbors = [(x, self.simMat[x0, x], r) for (x, r) in self.yr[y0]]

        est = self.means[x0]

        if not neighbors:
            return est # result will be just the baseline

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x:x[1], reverse=True)

        # compute weighted average
        sumSim = sumRatings = 0
        for (nb, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sumSim += sim
                sumRatings += sim * (r - self.means[nb])

        try:
            est += sumRatings / sumSim
        except ZeroDivisionError:
            pass # return mean

        return est

class AlgoWithBaseline(Algo):
    """Abstract class for algos that need a baseline"""
    def __init__(self, trainingData, itemBased, method, **kwargs):
        super().__init__(trainingData, itemBased, **kwargs)

        #compute users and items biases
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
            for x, y, r in self.iterAllRatings():
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
            for y in range(self.nY):
                devY = sum(r - self.meanRatings - self.xBiases[x] for (x, r) in self.yr[y])
                self.yBiases[y] = devY / (self.reg_y + len(self.yr[y]))

            self.xBiases = np.zeros(self.nX)
            for x in range(self.nX):
                devX = sum(r - self.meanRatings - self.yBiases[y] for (y, r) in self.xr[x])
                self.xBiases[x] = devX / (self.reg_x + len(self.xr[x]))

    def getBaseline(self, x, y):
        return self.meanRatings + self.xBiases[x] + self.yBiases[y]


class AlgoBaselineOnly(AlgoWithBaseline):
    """ Algo using only baseline"""

    def __init__(self, trainingData, itemBased=False, method='als', **kwargs):
        super().__init__(trainingData, itemBased, method=method)
        self.infos['name'] = 'algoBaselineOnly'

    def estimate(self, u0, i0):
        x0, y0 = self.getx0y0(u0, i0)
        return self.getBaseline(x0, y0)
