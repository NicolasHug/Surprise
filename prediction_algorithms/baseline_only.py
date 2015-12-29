from .bases import AlgoWithBaseline


class BaselineOnly(AlgoWithBaseline):
    """ Algo using only baseline"""

    def __init__(self, trainingData, itemBased=False, method='als', **kwargs):
        super().__init__(trainingData, itemBased, method=method, **kwargs)
        self.infos['name'] = 'algoBaselineOnly'

    def estimate(self, u0, i0):
        x0, y0 = self.getx0y0(u0, i0)
        return self.getBaseline(x0, y0)
