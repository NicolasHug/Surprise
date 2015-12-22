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
    def __init__(self, rm, ur, ir, itemBased=False, withDump=False, rMin=1,
            rMax=5):

        # whether the algo will be based on users (basically means that the
        # similarities will be computed between users or between items)
        # if the algo is user based, x denotes a user and y an item
        # if the algo is item based, x denotes an item and y a user
        self.ub = not itemBased

        if self.ub:
            self.rm = rm.T # we take the transpose of the rating matrix
            self.lastXi = cmn.lastUi
            self.lastYi = cmn.lastIi
            self.xr = ur
            self.yr = ir
        else:
            self.lastXi = cmn.lastIi
            self.lastYi = cmn.lastUi
            self.rm = rm
            self.xr = ir
            self.yr = ur

        # boundaries of ratings interval (usually [1, 5] or [0, 1])
        self.rMin = rMin
        self.rMax = rMax
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
        self.infos['ur'] = ur # user ratings  dict
        self.infos['ir'] = ir # item ratings dict
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
            print(est, self.meanRatings)

        # clip estimate into [self.rMin, self.rMax]
        est = min(self.rMax, self.est)
        est = max(self.rMin, self.est)

        if output:
            if impossible:
                print(cmn.Col.FAIL + 'Impossible to predict' + cmn.Col.ENDC)
            err = abs(self.est - r0)
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

    def __init__(self, rm, ur, ir, **kwargs):
        super().__init__(rm, ur, ir)
        self.infos['name'] = 'random'

        # estimation of the distribution
        fqs = [0, 0, 0, 0, 0]
        for x in range(1, self.lastXi):
            for y in range(1, self.lastYi):
                if self.rm[x, y] > 0:
                    fqs[self.rm[x, y] - 1] += 1
        fqs = [fq/sum(fqs) for fq in fqs]
        self.distrib = rv_discrete(values=([1, 2, 3, 4, 5], fqs))

    def estimate(self, *_):
        self.est = self.distrib.rvs()

class AlgoUsingSim(Algo):
    """Abstract class for algos using a similarity measure
    sim parameter can be 'cos' or 'MSD' for mean squared difference"""
    def __init__(self, rm, ur, ir, itemBased, sim, **kwargs):
        super().__init__(rm, ur, ir, itemBased, **kwargs)

        self.infos['params']['sim'] = sim
        self.constructSimMat(sim) # we'll need the similiarities

    def constructSimMat(self, sim):
        """construct the simlarity matrix"""

        print("computing the similarity matrix...")
        self.simMat = np.zeros((self.lastXi + 1, self.lastXi + 1))
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

        for xi in range(1, self.lastXi + 1):
            self.simMat[xi, xi] = 1
            for xj in range(xi + 1, self.lastXi + 1):
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

        for xi in range(1, self.lastXi + 1):
            self.simMat[xi, xi] = 100 # completely arbitrary and useless anyway
            for xj in range(xi, self.lastXi + 1):
                if sqDiff[xi, xj] == 0:  # return number of common ys
                    self.simMat[xi, xj] = freq[xi, xj]
                else:  # return inverse of MSD
                    self.simMat[xi, xj] = freq[xi, xj] / sqDiff[xi, xj]

                self.simMat[xj, xi] = self.simMat[xi, xj]

    def constructMSDCloneSimMat(self):
        """compute the 'clone' mean squared difference similarity between all
        pairs of xs. Some properties as for MSDSim apply"""

        for xi in range(1, self.lastXi + 1):
            self.simMat[xi, xi] = 100 # completely arbitrary and useless anyway
            for xj in range(xi, self.lastXi + 1):
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

        for xi in range(1, self.lastXi + 1):
            self.simMat[xi, xi] = 1
            for xj in range(xi + 1, self.lastXi + 1):
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

    def __init__(self, rm, ur, ir, itemBased=False, sim='cos', k=40, **kwargs):
        super().__init__(rm, ur, ir, itemBased=itemBased, sim=sim)

        self.k = k

        self.infos['name'] = 'KNNBasic'
        self.infos['params']['similarity measure'] = sim
        self.infos['params']['k'] = self.k

    def estimate(self, u0, i0):
        x0, y0 = self.getx0y0(u0, i0)
        # list of (x, sim(x0, x)) for u having rated i0 or for m rated by x0
        simX0 = [(x, self.simMat[x0, x]) for x in range(1, self.lastXi + 1) if
            self.rm[x, y0] > 0]

        # if there is nobody on which predict the rating...
        if not simX0:
            raise PredictionImpossible

        # sort simX0 by similarity
        simX0 = sorted(simX0, key=lambda x:x[1], reverse=True)

        # let the KNN vote
        simNeighboors = [sim for (_, sim) in simX0[:self.k] if sim > 0]
        ratNeighboors = [self.rm[x, y0] for (x, sim) in simX0[:self.k]
                if sim > 0]
        try:
            self.est = np.average(ratNeighboors, weights=simNeighboors)
        except ZeroDivisionError:
            raise PredictionImpossible

class AlgoKNNWithMeans(AlgoUsingSim):
    """Basic collaborative filtering algorithm, taking into account the mean
    ratings of each user"""

    def __init__(self, rm, ur, ir, itemBased=False, sim='cos', k=40, **kwargs):
        super().__init__(rm, ur, ir, itemBased=itemBased, sim=sim)

        self.k = k

        self.infos['name'] = 'basicWithMeans'
        self.infos['params']['similarity measure'] = sim
        self.infos['params']['k'] = self.k

        self.means = np.zeros(self.lastXi + 1)
        for x, ratings in self.xr.items():
            self.means[x] = np.mean([r for (_, r) in self.xr[x]])

    def estimate(self, u0, i0):
        x0, y0 = self.getx0y0(u0, i0)
        # list of (x, sim(x0, x)) for u having rated i0 or for m rated by x0
        simX0 = [(x, self.simMat[x0, x]) for x in range(1, self.lastXi + 1) if
            self.rm[x, y0] > 0]

        self.est = self.means[x0]

        # if there is nobody on which predict the rating...
        if not simX0:
            return

        # sort simX0 by similarity
        simX0 = sorted(simX0, key=lambda x:x[1], reverse=True)

        # let the KNN vote
        simNeighboors = [sim for (_, sim) in simX0[:self.k] if sim > 0]
        ratNeighboors = [self.rm[x, y0] - self.means[x] for (x, sim) in
                simX0[:self.k] if sim > 0]
        try:
            self.est += np.average(ratNeighboors, weights=simNeighboors)
        except ZeroDivisionError:
            pass

class AlgoWithBaseline(Algo):
    """Abstract class for algos that need a baseline"""
    def __init__(self, rm, ur, ir, itemBased, method, **kwargs):
        super().__init__(rm, ur, ir, itemBased, **kwargs)

        #compute users and items biases
        # see from 5.2.1 of RS handbook

        self.xBiases = np.zeros(self.lastXi + 1)
        self.yBiases = np.zeros(self.lastYi + 1)

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
            self.yBiases = np.zeros(self.lastYi + 1)
            for y in range(1, self.lastYi + 1):
                devY = sum(r - self.meanRatings - self.xBiases[x] for (x, r) in self.yr[y])
                self.yBiases[y] = devY / (self.reg_y + len(self.yr[y]))

            self.xBiases = np.zeros(self.lastXi + 1)
            for x in range(1, self.lastXi + 1):
                devX = sum(r - self.meanRatings - self.yBiases[y] for (y, r) in self.xr[x])
                self.xBiases[x] = devX / (self.reg_x + len(self.xr[x]))

    def getBaseline(self, x, y):
        return self.meanRatings + self.xBiases[x] + self.yBiases[y]


class AlgoBaselineOnly(AlgoWithBaseline):
    """ Algo using only baseline"""

    def __init__(self, rm, ur, ir, itemBased=False, method='als', **kwargs):
        super().__init__(rm, ur, ir, itemBased, method=method)
        self.infos['name'] = 'algoBaselineOnly'

    def estimate(self, u0, i0):
        x0, y0 = self.getx0y0(u0, i0)
        self.est = self.getBaseline(x0, y0)

