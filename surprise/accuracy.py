"""
The :mod:`surprise.accuracy` module provides tools for computing accuracy
metrics on a set of predictions.

Available accuracy metrics:

.. autosummary::
    :nosignatures:

    rmse
    mse
    mae
    fcp
"""

from collections import defaultdict
import numpy as np


def rmse(predictions, verbose=True):
    """Compute RMSE (Root Mean Squared Error).

    .. math::
        \\text{RMSE} = \\sqrt{\\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2}.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Root Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse = np.mean([float((true_r - est)**2)
                   for (_, _, true_r, est, _) in predictions])
    rmse_ = np.sqrt(mse)

    if verbose:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_


def mse(predictions, verbose=True):
    """Compute MSE (Mean Squared Error).

    .. math::
        \\text{MSE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}(r_{ui} - \\hat{r}_{ui})^2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Squared Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mse_ = np.mean([float((true_r - est)**2)
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MSE: {0:1.4f}'.format(mse_))

    return mse_


def mae(predictions, verbose=True):
    """Compute MAE (Mean Absolute Error).

    .. math::
        \\text{MAE} = \\frac{1}{|\\hat{R}|} \\sum_{\\hat{r}_{ui} \in
        \\hat{R}}|r_{ui} - \\hat{r}_{ui}|

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Mean Absolute Error of predictions.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    mae_ = np.mean([float(abs(true_r - est))
                    for (_, _, true_r, est, _) in predictions])

    if verbose:
        print('MAE:  {0:1.4f}'.format(mae_))

    return mae_


def fcp(predictions, verbose=True):
    """Compute FCP (Fraction of Concordant Pairs).

    Computed as described in paper `Collaborative Filtering on Ordinal User
    Feedback <http://www.ijcai.org/Proceedings/13/Papers/449.pdf>`_ by Koren
    and Sill, section 5.2.

    Args:
        predictions (:obj:`list` of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`):
            A list of predictions, as returned by the :meth:`test()
            <surprise.prediction_algorithms.algo_base.AlgoBase.test>` method.
        verbose: If True, will print computed value. Default is ``True``.


    Returns:
        The Fraction of Concordant Pairs.

    Raises:
        ValueError: When ``predictions`` is empty.
    """

    if not predictions:
        raise ValueError('Prediction list is empty.')

    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for u0, _, r0, est, _ in predictions:
        predictions_u[u0].append((r0, est))

    for u0, preds in predictions_u.items():
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.mean(list(nc_u.values())) if nc_u else 0
    nd = np.mean(list(nd_u.values())) if nd_u else 0

    try:
        fcp = nc / (nc + nd)
    except ZeroDivisionError:
        raise ValueError('cannot compute fcp on this list of prediction. ' +
                         'Does every user have at least two predictions?')

    if verbose:
        print('FCP:  {0:1.4f}'.format(fcp))

    return fcp
