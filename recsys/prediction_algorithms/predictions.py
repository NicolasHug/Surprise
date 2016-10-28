"""
The :mod:`recsys.prediction_algorithms.predictions` module defines the
:class:`Prediction` named tuple and the :class:`PredictionImpossible`
exception.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import namedtuple


class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible.

    When raised, the estimation :math:`\hat{r}_{ui}` is set to the global mean
    of all ratings :math:`\mu`.
    """

    pass


class Prediction(namedtuple('Prediction',
                            ['uid', 'iid', 'r0', 'est', 'details'])):
    """A named tuple for storing the results of a prediction.

    It's wrapped in a class, but only for documentation and printing purposes.

    Args:
        uid: The (inner) user id. See :ref:`this note<raw_inner_note>`.
        iid: The (inner) item id. See :ref:`this note<raw_inner_note>`.
        r0: The true rating :math:`r_{ui}`.
        est: The estimated rating :math:`\\hat{r}_{ui}`.
        details (dict): Stores additional details about the prediction that
            might be useful for later analysis.
    """

    __slots__ = ()  # for memory saving purpose.

    def __str__(self):
        s = 'user: {uid:<10} '.format(uid=self.uid)
        s += 'item: {iid:<10} '.format(iid=self.iid)
        s += 'r = {r0:1.2f}   '.format(r0=self.r0)
        s += 'est = {est:1.2f}   '.format(est=self.est)
        s += str(self.details)

        return s
