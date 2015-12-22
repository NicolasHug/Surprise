import warnings
import numpy as np

import common as cmn

def err(p):
    """return the error between the expected rating and the estimated one"""
    return p['est'] - p['r0']

def meanCommonYs(p):
    """return the mean count of users (or items) rated in common for all
    the 3-tuples of prediction p"""
    return np.mean(p['3tuples'], 0)[3] if p['3tuples'] else 0

def solProp(p, r):
    """proportion of solution to analogical equation that are equal to r for
    all 3-tuples of the prediction"""
    if p['3tuples']:
        return sum((rd == r) for _, _, _, _, rd in p['3tuples'])/len(p['3tuples'])
    else :
        return 0

def getx0y0(p, infos):
    """return x0 and y0 based on the ub variable"""
    if infos['ub']:
        return p['u0'], p['i0']
    else:
        return p['i0'], p['u0']


def getCommonYs(t, p, infos):
    xa, xb, xc, _, _ = t
    x0, _ = getx0y0(p, infos)
    xr = infos['ur'] if infos['ub'] else infos['ir']
    rm = infos['rm']
    Yabc0 = [y for (y, r) in xr[xa] if rm[xb, y] and rm[xc, y] and rm[x0, y]]
    return Yabc0

def details(p, infos):
    """print details about a prediction"""
    def detailsRatings(x='u'):
        """print mean and count of ratings for a user or an item"""
        xr = infos['ur'] if x == 'u' else infos['ir']
        x0 = 'u0' if x == 'u' else 'i0'
        print("\tcount: {0:d}".format(len(xr[p[x0]])))
        s = "{0:1.4f}".format(np.mean(xr[p[x0]], 0)[1]) if xr[p[x0]] else ""
        print("\tmean :", s)

    # ids, true rating, etimation and error
    print("u0: {0:<3d}    i0: {1:<4d}   r0: {2}   est: {3:1.2f}"
        "   err: {4:-2.2f}".format(p['u0'], p['i0'], p['r0'], p['est'],
        err(p)))

     # was the prediction impossible ?
    print("Prediction impossible? -", p['wasImpossible'])

    # u0 and i0 ratings infos
    print("u0 ratings:")
    detailsRatings(x='u')
    print("i0 ratings:")
    detailsRatings(x='m')

    # if algo is analogy based, print info about the candidates triplets
    if '3tuples' in p:
        print("3-tuples:")
        print("\tcount: {0:d}".format(len(p['3tuples'])))
        print("\tmean of common ys : {0:2.0f}".format(meanCommonYs(p)))
        print("\tproportion of solutions among the candidate 3-tuples:")
        lineLenght = 50
        rm = infos['rm']
        for r in [1, 2, 3, 4, 5]:
            print('\t\tsol =', r, ': [', end="")
            prop = solProp(p, r)
            nFill = int(prop * lineLenght)
            print('X' * nFill + ' ' * (lineLenght - nFill), end="")
            print('] - {0:3.0f}%'.format(prop*100.))

        """
        x0, y0 = getx0y0(p, infos)
        print("\tdetails for 3-tuples:")
        for t in p['3tuples']:
            xa, xb, xc, _, rd = t
            rsol = rm[xc, y0] - rm[xa, y0] + rm[xb, y0]
            tvs = []
            for y in getCommonYs(t, p , infos):
                tva = cmn.tvA(rm[xa, y], rm[xb, y], rm[xc, y], rm[x0, y])
                tvs.append(tva)
                print("\t\t{0:4d}  {1:d}  {2:d}  {3:d}  {4:d}"
                "  {5:1.2f}".format(y, rm[xa, y], rm[xb, y], rm[xc, y], rm[x0,
                    y], tva))
            print("\t\tmean tvA = {0:1.2f}".format(np.mean(tvs)))
            print("\t\t{0:d}  {1:d}  {2:d}  {3:d}".format( rm[xa, y0], rm[xb,
            y0], rm[xc, y0], rsol))
            print("\t\t" + '-' * 16)
        """



def errorBetween(p, inf=0., sup=4.):
    """return true if abs(err) is between inf and sup (both included)"""
    return inf <= abs(err(p)) <= sup

def ratingsCountBetween(p, x='u', inf=0, sup=float('inf')):
    """return true if the number of rating for x0 ('u' or 'm') is between inf
    and sup (both included)"""
    xr = infos['ur'] if x == 'u' else infos['ir']
    x0 = 'u0' if x == 'u' else 'i0'
    return inf <= len(xr[p[x0]]) <= sup

def r0Between(p, inf=1, sup=5):
    """return true if r0 is between inf and sup (both included)"""
    return inf <= p['r0'] <= sup

def meanCommonXsBetween(p, inf=0, sup=float('inf')):
    """return true if the mean of common ratings is betewen inf and sup (both
    included)"""
    return inf <= meanCommonXs(p) <= sup

def printHist(preds, key):
    """print histogram for errors ('err'), r0 ('r0') or round of estimations
    ('est')"""
    lineLenght = 50
    if key == 'err':
        for inf in range(4):
            print(inf, '<= err < ', inf + 1, ': [', end="")
            propInterval = (sum(inf <= abs(err(p)) < inf + 1 for p in preds) /
                len(preds))
            nFill = int(propInterval * lineLenght)
            print('X' * nFill + ' ' * (lineLenght - nFill), end="")
            print('] - {0:3.0f}%'.format(propInterval*100.))
    else:
        for v in range(1, 6):
            print(key, '=', v, ': [', end="")
            propInterval = (sum(v == round(p[key]) for p in preds) /
                len(preds))
            nFill = int(propInterval * lineLenght)
            print('X' * nFill + ' ' * (lineLenght - nFill), end="")
            print('] - {0:3.0f}%'.format(propInterval*100.))


def secsToHMS(s):
    """convert seconds to h:m:s"""
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), s

def pmi(i, j, pij):
    """return the point-wise mutual information of two items based on their
    probabilities to be rated together"""
    try:
        return np.log2(pij[i, j] / (pij[i, i] * pij[j, j])) / (-np.log2(pij[i, j]))
    except RuntimeWarning:
        return 0


def measureSurprise(preds, infos):
    """measure surprise of recommendation using PMI (see Adomavicius paper)"""

    # pij is a matrix of probabilities.
    # p[i, j] with i /= j represents the probability for i and j to be rated
    # together
    # p[i, i] represents the probability for i to be rated
    pij = np.zeros((cmn.lastIi + 1, cmn.lastIi + 1))
    print("constructing the pij matrix...")
    for i in range(1, cmn.lastIi + 1):
        iratings = infos['ir'][i]
        if not iratings:
            continue
        ui, _ = zip(*iratings)
        ui = set(ui) # we need a set to use intersection
        for j in range(i + 1, cmn.lastIi + 1):
            jratings = infos['ir'][j]
            if not jratings:
                continue
            uj, _ = zip(*(jratings))
            uj = set(uj)
            pij[i, j] = len(ui.intersection(uj)) / cmn.lastUi
            pij[j, i] = pij[i, j]
        pij[i, i] = len(ui) / cmn.lastUi

    warnings.filterwarnings('error') # treat warning as exceptions

    # we measure surprise as max of PMI values or as their mean
    surprises = [] # list containing surprise measures of all predictions
    for p in filter(lambda p:p['est'] >= 4, preds):
        u0 = p['u0']
        i0 = p['i0']
        pmis = [pmi(i0, j, pij) for (j, _) in infos['ur'][u0]]
        surprises.append((max(pmis), np.mean(pmis)))

    print("mean of co-occurence surprise (max): "
        "{0:1.4f}".format(np.mean(surprises, 0)[0]))
    print("mean of co-occurence surprise (avg): "
        "{0:1.4f}".format(np.mean(surprises, 0)[1]))

def printCoverage(preds):

    # set of recommended items
    recItems = {p['i0'] for p in preds if p['est'] >= 4}
    # set of all items
    items = {p['i0'] for p in preds}

    print("coverage: {0:1.2f}".format(len(recItems) / len(items)))
