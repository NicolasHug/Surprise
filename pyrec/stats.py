import numpy as np

def get_rmse_mae(preds, output=True):
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

def get_FCP(preds, output=True):

    nc = 0
    nd = 0
    for _, _, r0i, esti, _ in preds:
        for _, _, r0j, estj, _ in preds:
            if esti > estj and r0i > r0j:
                nc += 1
            if esti >= estj and r0i < r0j:
                nd += 1

    fcp = nc / (nc + nd)
    print('FCP: {0:1.4f}'.format(fcp))

    return fcp
