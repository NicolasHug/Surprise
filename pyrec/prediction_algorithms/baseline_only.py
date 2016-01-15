"""
Algorithm predicting only the baseline.
"""

from .bases import AlgoWithBaseline


class BaselineOnly(AlgoWithBaseline):
    """Algorithm predicting the *baseline rating* for given user and item.

    :math:`\hat{r}_{ui} = b_{ui}`

    (see paper *Factor in the Neighbors: Scalable and
    Accurate Collaborative Filtering* by Yehuda Koren for details).

    """

    def __init__(self, user_based=True, method='als', **kwargs):

        super().__init__(user_based, method=method, **kwargs)
        self.infos['name'] = 'algoBaselineOnly'

    def train(self, trainset):

        super().train(trainset)

    def estimate(self, x0, y0):

        return self.get_baseline(x0, y0)
