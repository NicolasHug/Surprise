from scipy.stats import rv_discrete
from scipy.spatial.distance import cosine
import numpy as np
import random as rd

import common as cmn

class Algo:
    def __init__(self, rm, movieBased=False):
        self.ub = not movieBased

        if self.ub:
            self.rm = rm.T
            self.lastXi = cmn.lastUi
            self.lastYi = cmn.lastMi
        else:
            self.lastXi = cmn.lastMi
            self.lastYi = cmn.lastUi
            self.rm = rm

        self.est = 0
        self.preds = []
        self.mae = self.rmse = self.accRate = 0

    def constructSimMat(self):
        # open or precalculate the similarity matrix if it does not exist yet
        try:
            #print('Opening simFile')
            if self.ub:
                simFile = open('simCosUsers', 'rb')
                self.simMat = np.load(simFile)
            else:
                simFile = open('simCosMovies', 'rb')
                self.simMat = np.load(simFile)

        except IOError:
            print('Failed... Computation of similarities...')
            self.simMat = np.empty((self.lastXi + 1, self.lastXi + 1))
            for xi in range(1, self.lastXi + 1):
                for xj in range(1, self.lastXi + 1):
                    self.simMat[xi, xj] = self.simCos(xi, xj)
            if self.ub:
                simFile = open('simCosUsers', 'wb')
                np.save(simFile, self.simMat)
            else:
                simFile = open('simCosMovies', 'wb')
                np.save(simFile, self.simMat)

    def simCos(self, a, b):
        """ returns the similarity between two users or movies using cosine distance"""
        # movies rated by a and b or users having rated a and b
        Yab = [y for y in range(1, self.lastYi + 1) if self.rm[a, y] > 0 and
            self.rm[b, y] > 0]

        # need to have at least two movies in common
        if len(Yab) < 2:
            return 0

        # list of ratings of/by a and b
        aR = [self.rm[a, y] for y in Yab]
        bR = [self.rm[b, y] for y in Yab]

        return 1 - cosine(aR, bR)
        

    def getx0y0(self, u0, m0):
        if self.ub:
            return u0, m0
        else:
            return m0, u0

    def updatePreds(self, r0, output=True):
        self.preds.append((self.est, r0))
        
        if output:
            if self.est == 0:
                print(cmn.Col.FAIL + 'Impossible to predict' + cmn.Col.ENDC)
                self.est = 3 # default value
            if self.est == r0:
                print(cmn.Col.OKGREEN + 'OK' + cmn.Col.ENDC)
            else:
                print(cmn.Col.FAIL + 'KO ' + cmn.Col.ENDC + str(self.est))

    def makeStats(self, output=True):
        nOK = nKO = nImp = 0
            
        sumSqErr = 0
        sumAbsErr = 0

        for est, r0 in self.preds:
            if est == 0:
                nImp += 1
            else:
                sumSqErr += (r0 - est)**2
                sumAbsErr += abs(r0 - est)

                if est == r0:
                    nOK += 1
                else:
                    nKO += 1
        
        self.rmse = np.sqrt(sumSqErr / (nOK + nKO))
        self.mae = np.sqrt(sumAbsErr / (nOK + nKO))
        self.accRate = nOK / (nOK + nKO)

        if output:
            print('Nb impossible predictions:', nImp)
            print('RMSE:', self.rmse)
            print('MAE:', self.mae)
            print('Accuracy rate:', self.accRate)

class AlgoRandom(Algo):
    
    def __init__(self, rm):
        super().__init__(rm)

        # estimation of the distribution
        fqs = [0, 0, 0, 0, 0]
        for x in range(1, self.lastXi):
            for y in range(1, self.lastYi):
                if self.rm[x, y] > 0:
                    fqs[self.rm[x, y] - 1] += 1
        fqs = [fq/sum(fqs) for fq in fqs]
        self.distrib = rv_discrete(values=([1, 2, 3, 4, 5], fqs))


    def estimate(self, u0, m0):
        self.est = self.distrib.rvs()

class AlgoBasicCollaborative(Algo):

    def __init__(self, rm, movieBased=False):
        super().__init__(rm, movieBased)
        self.constructSimMat()

    def estimate(self, u0, m0):
        x0, y0 = self.getx0y0(u0, m0)
        # list of (x, sim(x0, x)) for u having rated m0 or for m rated by x0
        simX0 = [(x, self.simMat[x0, x]) for x in range(1, self.lastXi + 1) if
            self.rm[x, y0] > 0]

        # if there is nobody on which predict the rating...
        if not simX0:
            self.est = 0
            return

        simX0 = sorted(simX0, key=lambda x:x[1], reverse=True)

        # let the KNN vote
        k = 50
        simNeighboors = [sim for (_, sim) in simX0[:k]]
        ratNeighboors = [self.rm[x, y0] for (x, _) in simX0[:k]]
        try:
            self.est = int(round(np.average(ratNeighboors,
                weights=simNeighboors)))
        except ZeroDivisionError:
            self.est = 0


class AlgoConf(Algo):
    def __init__(self, rm, confMeasure, movieBased=False):
        super().__init__(rm, movieBased)
        self.constructSimMat()
        self.confMeasure = confMeasure

    def estimate(self, u0, m0):
        x0, y0 = self.getx0y0(u0, m0)

        confs = {}
        for rConf in [1, 2, 3, 4, 5]:
            self.rm[x0, y0] = rConf
            confGroup = self.getConfGroup(rConf, x0, y0, k=10)
            confs[rConf] = self.confMeasure(self, x0, y0, confGroup)
        self.rm[x0, y0] = 0

        # check if no conf was computable
        if all(conf == float('-inf') for conf in confs.values()):
            self.est = 0
        else:
            sortedConfs = sorted(confs.keys(), key=lambda x:confs[x], reverse=True)
            print(sortedConfs)
            self.est = sortedConfs[0]

    def getConfGroup(self, rConf, x0, y0, k=10):
        """ return the k most similar users/movies to u0/m0 such that rating =
        rConf
        """

        confGroup = [(x, self.simMat[x0, x]) for (x, r) in enumerate(self.rm[:,
            y0]) if r == rConf and x != x0]

        confGroup = sorted(confGroup, key=lambda x:x[1], reverse=True)
        if k == 0:
            k = len(confGroup)
        confGroup = [u for (u, _) in confGroup[:k]]

        return confGroup

    def maxIdHd(self, x0, y0, confGroup):
        if len(confGroup) < 3:
            return float('-inf')
            
        x0Y = [y for (y, r) in enumerate(self.rm[x0, :]) if r > 0]

        conf = 0
        nConfs = 0
        for xa in confGroup:
            for xb in confGroup:
                for xc in confGroup:
                    if xa != xc != xb and xa != xb:
                        for y in x0Y:
                            if (self.rm[xa, y] and self.rm[xb, y] and 
                                self.rm[xc, y]):
                                nConfs += 1
                                a = (self.rm[xa, y] - 1) / 4
                                b = (self.rm[xb, y] - 1) / 4
                                c = (self.rm[xc, y] - 1) / 4
                                d = (self.rm[x0, y] - 1) / 4
                                conf += max(cmn.idty(a, b, c, d), cmn.hd1(a, b, c, d))
        try:
            conf = conf / nConfs 
        except ZeroDivisionError:
            conf = float('-inf')
        return conf

    def idSymb(self, x0, confGroup):
        x0Y = [y for (y, r) in enumerate(self.rm[x0, :]) if r > 0]

        conf = 0
        nConfs = 0
        for xa in confGroup:
            for xb in confGroup:
                if xa != xb:
                    for y in x0Y:
                        if self.rm[xa, y] and  self.rm[xb, y]:
                            nConfs += 1
                            if (self.rm[xa, y] == self.rm[x0, y] or 
                                self.rm[xb, y] == self.rm[x0, y]):
                                conf += 1
        try:
            conf = conf / nConfs 
        except ZeroDivisionError:
            conf = float('-inf')
        return conf

class AlgoAnalogy(Algo):
    def __init__(self, rm, movieBased=False):
        super().__init__(rm, movieBased)
        self.constructSimMat()

    def estimate(self, u0, m0):
        x0, y0 = self.getx0y0(u0, m0)

        # list (x, r) for y0
        y0Xs = [(x, r) for (x, r) in enumerate(self.rm[:, y0]) if r > 0]
        # list of y for x0
        x0Ys = [y for (y, r) in enumerate(self.rm[x0, :]) if r > 0]

        if not y0Xs:
            self.est = 0
            return

        sols = []
        for i in range(1000):
            xa, ra = rd.choice(y0Xs)
            xb, rb = rd.choice(y0Xs)
            xc, rc = rd.choice(y0Xs)
            if xa != xb != xc and xa != xc and self.isSolvable(ra, rb, rc):
                tv = self.getTvVector(xa, xb, xc, x0, x0Ys)
                if tv:
                    sols.append((cmn.solveAstar(ra, rb, rc), np.mean(tv)))

        ratings = [r for (r, _) in sols]
        weights = [w for (_, w) in sols]

        if not ratings or set(weights) == {0}:
            self.est = 0
            return
        self.est = int(round(np.average(ratings, weights=weights)))


                        

    def getTvVector(self, xa, xb, xc, x0, x0Ys):
        tv = []
        for y in x0Ys:
            if self.rm[xa, y] and self.rm[xb, y] and self.rm[xc, y]:
                rabc0 = [self.rm[xa, y], self.rm[xb, y], self.rm[xc, y],
                    self.rm[x0, y]]
                rabc0 = [(r - 1) / 4 for r in rabc0]
                #tv.append(cmn.tvAStar(*rabc0))
                tv.append(cmn.tvA(*rabc0))
        return tv

    def estimate1(self, u0, m0):
        x0, y0 = self.getx0y0(u0, m0)

        # list (x, r) for y0
        y0Xs = [(x, r) for (x, r) in enumerate(self.rm[:, y0]) if r > 0]
        # list of y for x0
        x0Ys = [y for (y, r) in enumerate(self.rm[x0, :]) if r > 0]

        if not y0Xs:
            self.est = 3
            return

        triplets = []
        for i in range(1000):
            xa, ra = rd.choice(y0Xs)
            xb, rb = rd.choice(y0Xs)
            xc, rc = rd.choice(y0Xs)
            if xa != xb != xc and xa != xc and self.isSolvable(ra, rb, rc):
                triplets.append((xa, xb, xc))

        if not triplets:
            self.est = 3
            return

        d = dict()
        for xa, xb, xc in triplets:
            tv = []
            for y in x0Ys:
                if self.rm[xa, y] and self.rm[xb, y] and self.rm[xc, y]:
                    rabc0 = [self.rm[xa, y], self.rm[xb, y], self.rm[xc, y],
                    self.rm[x0, y]]
                    rabc0 = [(r - 1) / 4 for r in rabc0]
                    #tv.append(cmn.tvAStar(*rabc0))
                    tv.append(cmn.tvA(*rabc0))
            d[(xa, xb, xc)] = tv



        longestTv = max(len(l) for l in d.values())
        print('longest tv vector size:', longestTv)
        for t in d.keys():
            d[t].extend([0.5] * (longestTv - len(d[t]) ))
            d[t].sort(reverse=True)

        # sort triplets with leximax
        bestTriplets = list(d.keys())
        bestTriplets = sorted(d.keys(), key=lambda x:d[x], reverse=True)

        ests = []
        for (xa, xb, xc) in bestTriplets[:11]:
            ests.append(int(cmn.solveAstar(self.rm[xa, y0], self.rm[xb, y0],
            self.rm[xc, y0])))

        self.est = int(round(np.average(ests)))


    def isSolvable(self, ra, rb, rc):
        return ra == rb or ra == rc


class AlgoGilles(Algo):
    def __init__(self, rm, movieBased=False):
        super().__init__(rm, movieBased)

    def estimate(self, u0, m0):
        x0, y0 = self.getx0y0(u0, m0)

        y0Xs = [x for x in range(1, self.lastXi) if self.rm[x, y0] > 0]

        if not y0Xs :
            self.est = 0
            return

        candidates= []
        for i in range(1000):
            xa = rd.choice(y0Xs)
            xb = rd.choice(y0Xs)
            xc = rd.choice(y0Xs)
            if xa != xb != xc and xa != xc and self.isSolvable(xa, xb, xc, y0):
                (nYabc0, nrm) = self.getParall(xa, xb, xc, x0)
                if nrm < 1.5 * np.sqrt(nYabc0):
                    candidates.append((self.solve(xa, xb, xc, y0), nrm, nYabc0))

        if candidates:
            ratings = [r for (r, _, _) in candidates]
            norms = [1/(nrm + 1) for (_, nrm, _) in candidates]
            nYs = [nY for (_, _, nY) in candidates]

            """
            self.est = int(round(np.average(ratings, weights=norms)))
            self.est = int(round(np.average(ratings, weights=nYs)))
            """
            self.est = int(round(np.average(ratings)))
        else:
            self.est = 0

    def isSolvable(self, xa, xb, xc, y0):
        return (self.rm[xa, y0] == self.rm[xb, y0] or self.rm[xa, y0] ==
        self.rm[xc, y0])

    """
    def solve(self, xa, xb, xc, y0):
        return (self.rm[xb, y0] if self.rm[xa, y0] == self.rm[xc, y0] else
        self.rm[xc, y0])

    """
    def solve(self, xa, xb, xc, y0):
        return self.rm[xc, y0] - self.rm[xa, y0] + self.rm[xb, y0]


    """
    def isSolvable(self, xa, xb, xc, y0):
        return (0 < self.solve(xa, xb, xc, y0) < 6)
    """

    def getParall(self, xa, xb, xc, x0):
        Yabc0 = [y for y in range(1, self.lastYi) if self.rm[xa, y] and
            self.rm[xb, y] and self.rm[xc, y] and self.rm[x0, y]]

        if not Yabc0:
            return 0, float('inf')

        xaRs = np.array([self.rm[xa, y] for y in Yabc0])
        xbRs = np.array([self.rm[xb, y] for y in Yabc0])
        xcRs = np.array([self.rm[xc, y] for y in Yabc0])
        x0Rs = np.array([self.rm[x0, y] for y in Yabc0])
        nrm = np.linalg.norm((xaRs - xbRs) - (xcRs - x0Rs))
        return len(Yabc0), nrm
