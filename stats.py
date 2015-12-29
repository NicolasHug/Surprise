import numpy as np


def computeStats(preds, output=True):
    """compute some statistics (RMSE, coverage...) on a list of predictions"""

    nImp = 0
    sumSqErr = 0
    sumAbsErr = 0

    for _, _, r0, est, imp in preds:

        sumSqErr += (r0 - est)**2
        sumAbsErr += abs(r0 - est)
        nImp += imp

    rmse = np.sqrt(sumSqErr / len(preds))
    mae = np.sqrt(sumAbsErr / len(preds))

    if output:
        print('Nb impossible predictions:', nImp)
        print('RMSE: {0:1.4f}'.format(rmse))
        print('MAE: {0:1.4f}'.format(mae))

    return rmse
