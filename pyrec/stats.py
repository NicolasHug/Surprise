import numpy as np

def compute_stats(preds, output=True):
    """compute some statistics (RMSE, coverage...) on a list of predictions"""

    nImp = 0
    sum_sq_err = 0
    sum_abs_err = 0

    for _, _, r0, est, imp in preds:

        sum_sq_err += (r0 - est)**2
        sum_abs_err += abs(r0 - est)
        nImp += imp

    rmse = np.sqrt(sum_sq_err / len(preds))
    mae = np.sqrt(sum_abs_err / len(preds))

    if output:
        print('Nb impossible predictions:', nImp)
        print('RMSE: {0:1.4f}'.format(rmse))
        print('MAE: {0:1.4f}'.format(mae))

    return rmse, mae
