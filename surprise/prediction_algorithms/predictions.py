"""
The :mod:`surprise.prediction_algorithms.predictions` module defines the
:class:`Prediction` named tuple and the :class:`PredictionImpossible`
exception.
"""


from collections import namedtuple


class PredictionImpossible(Exception):
    r"""Exception raised when a prediction is impossible.

    When raised, the estimation :math:`\hat{r}_{ui}` is set to the global mean
    of all ratings :math:`\mu`.
    """

    pass


class Prediction(namedtuple("Prediction", ["uid", "iid", "r_ui", "est", "details"])):
    """A named tuple for storing the results of a prediction.

    It's wrapped in a class, but only for documentation and printing purposes.

    Args:
        uid: The (raw) user id. See :ref:`this note<raw_inner_note>`.
        iid: The (raw) item id. See :ref:`this note<raw_inner_note>`.
        r_ui(float): The true rating :math:`r_{ui}`.
        est(float): The estimated rating :math:`\\hat{r}_{ui}`.
        details (dict): Stores additional details about the prediction that
            might be useful for later analysis.
    """

    __slots__ = ()  # for memory saving purpose.

    def __str__(self):
        s = f"user: {self.uid:<10} "
        s += f"item: {self.iid:<10} "
        if self.r_ui is not None:
            s += f"r_ui = {self.r_ui:1.2f}   "
        else:
            s += "r_ui = None   "
        s += f"est = {self.est:1.2f}   "
        s += str(self.details)

        return s
