"""
Algorithm predicting only the baseline.
"""

from .bases import AlgoBase


class BaselineOnly(AlgoBase):
    """Algorithm predicting the *baseline rating* for given user and item.

    :math:`\hat{r}_{ui} = b_{ui}`

    (see paper *Factor in the Neighbors: Scalable and
    Accurate Collaborative Filtering* by Yehuda Koren for details).

    """

    def __init__(self, **kwargs):

        AlgoBase.__init__(self, **kwargs)

    def train(self, trainset):

        AlgoBase.train(self, trainset)
        self.compute_baselines()

    def estimate(self, x0, y0):

        return self.get_baseline(x0, y0)
