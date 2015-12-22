import numpy as np

lastIi = 1682 # last item index for u1.base
lastUi = 943 # last user index for u1.base

class Col:
    """A class for adding color in the term output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def computeStats(preds, output=True):
    """compute some statistics (RMSE, coverage...) on a list of predictions"""

    if not preds:
        print("looks like there's no prediction...")
        return

    nOK = nKO = nImp = 0

    sumSqErr = 0
    sumAbsErr = 0

    nRecoOK = nRecoKO = 0


    threshold = 4. # we recommend m to u iff estimation >= threshold

    for p in preds:

        sumSqErr += (p['r0'] - p['est'])**2
        sumAbsErr += abs(p['r0'] - p['est'])

        if p['est'] >= threshold: # we recommend m to u
            if p['r0'] >= threshold: # we did well
                nRecoOK += 1
            else: # we shouldn't have...
                nRecoKO += 1

        if p['est'] == p['r0']:
            nOK += 1
        else:
            nKO += 1
        if p['wasImpossible']:
            nImp += 1

    rmse = np.sqrt(sumSqErr / (nOK + nKO))
    mae = np.sqrt(sumAbsErr / (nOK + nKO))
    accRate = nOK / (nOK + nKO)
    precision = 0#nRecoOK / (nRecoOK + nRecoKO)
    recall = nRecoOK / sum(True for p in preds if p['r0'] >= threshold)

    if output:
        print('Nb impossible predictions:', nImp)
        print('RMSE: {0:1.4f}'.format(rmse))
        print('MAE: {0:1.4f}'.format(mae))
        print('sample size:', len(preds))
        print('Accuracy rate: {0:1.4f}'.format(accRate))
        print('Precision: {0:1.2f}'.format(precision))
        print('recall: {0:1.2f}'.format(recall))

    return rmse
