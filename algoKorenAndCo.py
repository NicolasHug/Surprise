from algo import *

class AlgoKNNBaseline(AlgoWithBaseline, AlgoUsingSim):
    """ Algo baseline AND deviation from baseline of the neighbors
        simlarity measure = cos"""
    def __init__(self, trainingData, itemBased=False, method='als', sim='cos',
            k=40, **kwargs):
        super().__init__(trainingData, itemBased, method=method, sim=sim)

        self.k = k
        self.infos['name'] = 'neighborhoodWithBaseline'
        self.infos['params']['k'] = self.k

    def estimate(self, u0, i0):
        x0, y0 = self.getx0y0(u0, i0)
        est = self.getBaseline(x0, y0)

        neighbors = [(x, self.simMat[x0, x], r) for (x, r) in self.yr[y0]]

        if not neighbors:
            return est # result will be just the baseline

        # sort neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x:x[1], reverse=True)

        # compute weighted average
        sumSim = sumRatings = 0
        for (nb, sim, r) in neighbors[:self.k]:
            if sim > 0:
                sumSim += sim
                sumRatings += sim * (r - self.getBaseline(nb, y0))

        try:
            est += sumRatings / sumSim
        except ZeroDivisionError:
            pass # just baseline again

        return est

class AlgoKNNBelkor(AlgoWithBaseline):
    """ KNN learning interpolating weights from the training data. see 5.1.1
    from reco system handbook"""
    def __init__(self, trainingData, itemBased=False, method='opt', **kwargs):
        super().__init__(trainingData, itemBased, method=method)
        self.weights = np.zeros((self.nX, self.nX),
        dtype='double')

        nIter = 20
        gamma = 0.005
        lambda10 = 0.002

        self.infos['name'] = 'KNNBellkor'

        for i in range(nIter):
            print("optimizing...", nIter - i, "iteration left")
            for x, y, rxy in self.allRatings:
                est = sum((r - self.getBaseline(x2, y)) *
                    self.weights[x, x2] for (x2, r) in self.yr[y])
                est /= np.sqrt(len(self.yr[y]))
                est += self.meanRatings + self.xBiases[x] + self.yBiases[y]

                err = rxy - est

                # update x bias
                self.xBiases[x] += gamma * (err - lambda10 *
                    self.xBiases[x])

                # update y bias
                self.yBiases[y] += gamma * (err - lambda10 *
                    self.yBiases[y])

                # update weights
                for x2, rx2y in self.yr[y]:
                    bx2y = self.getBaseline(x2, y)
                    wxx2 = self.weights[x, x2]
                    self.weights[x, x2] += gamma * ((err * (rx2y -
                        bx2y)/np.sqrt(len(self.yr[y]))) - (lambda10 * wxx2))


    def estimate(self, u0, i0):
        x0, y0 = self.getx0y0(u0, i0)

        est = sum((r - self.getBaseline(x2, y0)) *
            self.weights[x0, x2] for (x2, r) in self.yr[y0])
        est /= np.sqrt(len(self.yr[y0]))
        est += self.getBaseline(x0, y0)

        return est

class AlgoFactors(Algo):
    """Algo using latent factors. Implem heavily inspired by
    https://github.com/aaw/IncrementalSVD.jl"""
    def __init__(self, trainingData, itemBased=False, **kwargs):
        super().__init__(trainingData, itemBased)
        self.infos['name'] = 'algoLatentFactors'

        nFactors = 50 # number of factors
        nIter = 10
        self.px = np.ones((self.nX, nFactors)) * 0.1
        self.qy = np.ones((self.nY, nFactors)) * 0.1
        residuals = []

        lambda4 = 0.02 # regularization extent
        gamma = 0.005 # learning rate

        self.infos['params']['nFactors'] = nFactors
        self.infos['params']['reguParam'] = lambda4
        self.infos['params']['learningRate'] = gamma
        self.infos['params']['nIter'] = nIter

        ratings = []
        for x, y, val in self.allRatings:
            ratings.append(((x, y, val), [val, 0., 0.]))


        for f in range(nFactors):
            print(f)
            errors = [0., float('inf'), float('inf')]
            for i in range(nIter):
                for (x, y, val), (res) in ratings:
                    yF = self.qy[y, f] # value of feature f for y
                    xF = self.px[x, f] # value of feature f for x
                    res[1] = res[0] - yF * xF
                    errDiff = res[2] - res[1]
                    errors[0] += errDiff**2
                    res[2] = res[1]
                    self.qy[y, f] += gamma * (res[1] * xF - lambda4 * yF)
                    self.px[x, f] += gamma * (res[1] * yF - lambda4 * xF)
                errors = [0., errors[0], errors[1]]
            for _, (res) in ratings:
                res[0] = res[1]
                res[2] = 0.

    def estimate(self, u0, i0):
        x0, y0 = self.getx0y0(u0, i0)



        est = np.dot(self.px[x0, :], self.qy[y0, :])
        return est

