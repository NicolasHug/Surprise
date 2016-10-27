"""
The :mod:`recsys.prediction_algorithms.predictions` module defines the
:class:`Prediction` named tuple and the :class:`PredictionImpossible`
exception.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import namedtuple


class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible."""
    pass

class Prediction(namedtuple('Prediction',
                            ['uid', 'iid', 'r0', 'est', 'details'])):
    """A named tuple for storing the results of a prediction.

    It's wrapped in a class, but only for documentation purposes.

    Args:
        uid: The (inner) user id.
        iid: The (inner) item id.
        r0: The true rating :math:`r_{ui}`.
        est: The estimated rating :math:`\\hat{r}_{ui}`.
        details (dict): Stores additional details about the prediction that
            might be useful for later analysis.
        """

