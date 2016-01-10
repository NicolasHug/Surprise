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

    def __init__(self, training_data, item_based=False, method='als', **kwargs):
        super().__init__(training_data, item_based, method=method, **kwargs)
        self.infos['name'] = 'algoBaselineOnly'

    def estimate(self, u0, i0):
        x0, y0 = self.getx0y0(u0, i0)
        return self.get_baseline(x0, y0)
