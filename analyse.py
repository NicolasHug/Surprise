#!/usr/bin/python3

import pickle
import sys
import os

import numpy as np

import common as c

def err(p):
    """return the error between the expected rating and the estimated one"""
    return p['est'] - p['r0']

def meanCommonXs(p):
    """return the mean count of users (or movies) rated in common for all
    the 3-tuples of prediction p"""
    return np.mean(p['3tuples'], 0)[3] if p['3tuples'] else 0
    
def correctSolProp(p):
    """proportion of solution to analogical equation that were correct for all
    3-tuples of the prediction"""
    if p['3tuples']:
        return sum((rd == p['r0']) for _, _, _, _, rd in p['3tuples'])/len(p['3tuples'])
    else :
        return 0

def details(p):
    """print details about a prediction"""
    def detailsRatings(x='u'):
        """print mean and count of ratings for a user or a movie"""
        xr = infos['ur'] if x == 'u' else infos['mr']
        x0 = 'u0' if x == 'u' else 'm0'
        print("\tcount: {0:d}".format(len(xr[p[x0]])))
        s = "{0:1.4f}".format(np.mean(xr[p[x0]], 0)[1]) if xr[p[x0]] else ""
        print("\tmean :", s)

    # ids, true rating, etimation and error
    print("u0: {0:<3d}    m0: {1:<4d}   r0: {2}   est: {3:1.2f}"
        "   err: {4:-2.2f}".format(p['u0'], p['m0'], p['r0'], p['est'],
        err(p)))

     # was the prediction impossible ?
    print("Prediction impossible? -", p['wasImpossible'])

    # u0 and m0 ratings infos
    print("u0 ratings:")
    detailsRatings(x='u')
    print("m0 ratings:")
    detailsRatings(x='m')

    # if algo is analogy based, print info about the candidates triplets
    if '3tuples' in p:
        print("3-tuples:")
        print("\tcount: {0:d}".format(len(p['3tuples'])))
        print("\tmean of common xs : {0:2.0f}".format(meanCommonXs(p)))
        print("\tcorrect solution: {0:2.0f}%".format(correctSolProp(p)*100.))


def errorBetween(p, inf=0., sup=4.):
    """return true if abs(err) is between inf and sup (both included)"""
    return inf <= abs(err(p)) <= sup

def ratingsCountBetween(p, x='u', inf=0, sup=float('inf')):
    """return true if the number of rating for x0 ('u' or 'm') is between inf
    and sup (both included)"""
    xr = infos['ur'] if x == 'u' else infos['mr']
    x0 = 'u0' if x == 'u' else 'm0'
    return inf <= len(xr[p[x0]]) <= sup

def r0Between(p, inf=1, sup=5):
    """return true if r0 is between inf and sup (both included)"""
    return inf <= p['r0'] <= sup

def meanCommonXsBetween(p, inf=0, sup=float('inf')):
    """return true if the mean of common ratings is betewen inf and sup (both
    included)"""
    return inf <= meanCommonXs(p) <= sup

def printHist(preds, key):
    """print histogram for errors ('err'), r0 ('r0') or estimations ('est')"""
    lineLenght = 50
    if key == 'err':
        for inf in range(4):
            print(inf, '<= err < ', inf + 1, ': [', end="")
            propInterval = (sum(inf <= abs(err(p)) < inf + 1 for p in preds) /
                len(preds))
            nFill = int(propInterval * lineLenght)
            print('X' * nFill + ' ' * (lineLenght - nFill), end="")
            print('] - {0:02.0f}%'.format(propInterval*100.))
    else: 
        for v in range(1, 6):
            print(key,'=', v, ': [', end="")
            propInterval = sum(p[key] == v for p in preds) / len(preds)
            nFill = int(propInterval * lineLenght)
            print('X' * nFill + ' ' * (lineLenght - nFill), end="")
            print('] - {0:02.0f}%'.format(propInterval*100.))

def secsToHMS(s):
    """convert seconds to h:m:s"""
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), s



# by default, open the last created file in the dumps directory
if len(sys.argv) < 2:
    for dirname, dirnames, filenames in os.walk('./dumps'):
        dumpFile = max([f for f in filenames], 
            key=lambda x:os.path.getctime(os.path.join(dirname, x)))
        dumpFile = os.path.join(dirname, dumpFile)
# if file name is passed as argument, chose it instead
else:
    dumpFile=sys.argv[1]

print('File:', dumpFile)

infos = pickle.load(open(dumpFile, 'rb'))

# print name and specific parameters
print('Algo name:', infos['name'])
print('Tests count:', len(infos['preds']))
for k, v in infos['params'].items():
    print(k + ':', v)


# requirements that a prediction needs to fit
requirements = (lambda p: 
     errorBetween(p, inf=3))

# list with all estimations fitting the previously defined requirements
# (list and not iterator because we may need to use it more than once)
interestingPreds= list(filter(requirements, infos['preds']))

# keep only the k highest or lowest values for ratings count of u0
"""
k = 5
interestingPreds.sort(key=lambda p:len(infos['ur'][p['u0']]))
interestingPreds = interestingPreds[-10:] # top K
#interestingPreds = interestingPreds[:10] # bottom K
"""


# print details for predictions we are interested in
"""
print('-' * 52)
for p in interestingPreds:
    details(p)
    print('-' * 52)
"""

print('-' * 52)
# print RMSE & Co for these predictions
c.printStats(interestingPreds)

    
# print proportion of absolute errors
print('-' * 10, "\n" + "Proportions of absolute errors among int. preds:")
printHist(interestingPreds, key='err')
print('-' * 10, "\n" + "Proportions of absolute errors among all preds:")
printHist(infos['preds'], key='err')

# print propotion of estimation values
print('-' * 52, "\n" + "Proportions of estimation values in int. preds:")
printHist(interestingPreds, key='est')
print('-' * 10, "\n" + "Proportions of estimation values in all preds:")
printHist(infos['preds'], key='est')

# print propotion of estimation values
print('-' * 52, "\n" + "Proportions of r0 values in int. preds:")
printHist(interestingPreds, key='r0')
print('-' * 10, "\n" + "Proportions of r0 values in all preds:")
printHist(infos['preds'], key='r0')

#print time info
print('-' * 52)
print("training time: "
    "{0:02d}h{1:02d}m{2:2.2f}s".format(*secsToHMS(infos['trainingTime'])))
print("testing time : "
    "{0:02d}h{1:02d}m{2:2.2f}s".format(*secsToHMS(infos['testingTime'])))
