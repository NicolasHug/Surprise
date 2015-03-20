import numpy as np

lastMi = 1682 # last movie index for u1.base
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

def printStats(preds):
    """compute some statistics (RMSE) on a list of predictions"""

    if not preds:
        print("looks like there's no prediction...")
        return

    nOK = nKO = nImp = 0
        
    sumSqErr = 0
    sumAbsErr = 0


    for p in preds:

        sumSqErr += (p['r0'] - p['est'])**2
        sumAbsErr += abs(p['r0'] - p['est'])

        if p['est'] == p['r0']:
            nOK += 1
        else:
            nKO += 1
        if p['wasImpossible']:
            nImp += 1

    rmse = np.sqrt(sumSqErr / (nOK + nKO))
    mae = np.sqrt(sumAbsErr / (nOK + nKO))
    accRate = nOK / (nOK + nKO)

    print('Nb impossible predictions:', nImp)
    print('RMSE:', rmse)
    print('MAE:', mae)
    print('sample size:', len(preds))
    print('Accuracy rate:', accRate)

def tvA(ra, rb, rc, rd):
    """return the truth value of A(ra, rb, rc, rd)"""

    # map ratings into [0, 1]
    ra = (ra-1)/4.; rb = (rb-1)/4.; rc = (rc-1)/4.; rd = (rd-1)/4.; 
    if (ra >= rb and rc >= rd) or (ra <= rb and rc <= rd):
        return 1 - abs((ra-rb) - (rc-rd))
    else:
        return 1 - max(abs(ra-rb), abs(rc-rd))
