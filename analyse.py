#!/usr/bin/python3

import pickle
import sys
import os

import numpy as np

import common as c

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

print('-' * 10)
print('-' * 10)

def err(p):
    """return the error between the expected rating and the estimated one"""
    return p['est'] - p['r0']

def meanCommonXs(p):
    """return the mean count of users (or movies) rated in common for all
    the 3-tuples of prediction p"""
    return np.mean(p['3tuples'], 0)[3] if p['3tuples'] else 0
    
    

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
        print("\tmean of common xs :{0:3.0f}".format(meanCommonXs(p)))


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
    return inf <= meanCommonXs(p) <= sup

# requirements that a prediction needs to fit
requirements = (lambda p: 
     meanCommonXsBetween(p, sup=5))

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
for p in interestingPreds:
    details(p)
    print('-' * 52)

# print RMSE & Co for these predictions
c.printStats(interestingPreds)

def secsToHMS(s):
    """convert seconds to h:m:s"""
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return int(h), int(m), s

print('-' * 10)
print("training time: "
    "{0:02d}h{1:02d}m{2:2.2f}s".format(*secsToHMS(infos['trainingTime'])))
print("testing time : "
    "{0:02d}h{1:02d}m{2:2.2f}s".format(*secsToHMS(infos['testingTime'])))

