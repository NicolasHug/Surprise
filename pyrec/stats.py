from statistics import mean
from math import sqrt
from collections import defaultdict

def rmse(predictions, output=True):
    """Compute RMSE (Root Mean Squared Error) on a list of predictions"""

    mse = mean(float((true_r - est)**2) for (_, _, true_r, est, _) in predictions)
    rmse_ = sqrt(mse)

    if output:
        print('RMSE: {0:1.4f}'.format(rmse_))

    return rmse_

def mae(predictions, output=True):
    """Compute MAE (Mean Absolute Error) on a list of predictions"""

    mae_ = mean(float(abs(true_r - est)) for (_, _, true_r, est, _) in predictions)

    if output:
        print('MAE: {0:1.4f}'.format(mae_))

    return mae_

def fcp(predictions, output=True):
    """Compute FCP (Fraction of Concordant Pairs) on a list of preditions"""

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

    nc = mean(nc_u.values())
    nd = mean(nd_u.values())

    fcp = nc / (nc + nd)

    if output:
        print('FCP: {0:1.4f}'.format(fcp))

    return fcp

