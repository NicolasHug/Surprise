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

def details(p):
    """print details on an estimation"""
    # ids, true rating, etimation and error
    print("u0: {0:<3d}    m0: {1:<4d}   r0: {2}   est: {3:1.2f}"
        "   err: {4:-2.2f}".format(p['u0'], p['m0'], p['r0'], p['est'],
        err(p)))
    # u0 ratings mean and count
    print("u0 ratings:")
    print("\tcount: {0:d}".format(len(infos['ur'][p['u0']])))
    print("\tmean : {0:1.4f}".format(np.mean(infos['ur'][p['u0']], 0)[1]))
    # m0 ratings mean and count
    print("m0 ratings:")
    print("\tcount: {0:d}".format(len(infos['mr'][p['m0']])))
    if infos['mr'][p['m0']]:
        print("\tmean : {0:1.4f}".format(np.mean(infos['mr'][p['m0']], 0)[1]))
    else:
        print("\tmean : no f***ing way!")
        

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

# requirements that a prediction needs to fit
requirements = (lambda p: 
    r0Between(p, sup=1))

# list with all estimations fitting the previously defined requirements
# (list and not iterator because we may need to use it more than once)
interstingPreds= list(filter(requirements, infos['preds']))


# print details for predictions we are interested in
for p in interstingPreds:
    details(p)
    print('-' * 52)

# print RMSE & Co for these predictions
c.printStats(interstingPreds)
