#!/usr/bin/python3

import pickle
import sys
import os

import numpy as np

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
print('Tests count:', len(infos['ests']))
for k, v in infos['params'].items():
    print(k + ':', v)

print('-' * 10)
print('-' * 10)

def err(e):
    """return the error between the expected rating and the estimated one"""
    return e['est'] - e['r0']

def details(e):
    """print details on an estimation"""
    # ids, true rating, etimation and error
    print("u0: {0:<3d}    m0: {1:<4d}   r0: {2}   est: {3:1.2f}"
        "   err: {4:-2.2f}".format(e['u0'], e['m0'], e['r0'], e['est'],
        err(e)))
    # u0 ratings mean and count
    print("u0 ratings:")
    print("\tcount: {0:d}".format(len(infos['ur'][e['u0']])))
    print("\tmean : {0:1.4f}".format(np.mean(infos['ur'][e['u0']], 0)[1]))
    # m0 ratings mean and count
    print("m0 ratings:")
    print("\tcount: {0:d}".format(len(infos['mr'][e['m0']])))
    if infos['mr'][e['m0']]:
        print("\tmean : {0:1.4f}".format(np.mean(infos['mr'][e['m0']], 0)[1]))
    else:
        print("\tmean : no f***ing way!")
        

def errorBetween(e, inf=0., sup=4.):
    """return true if abs(err) is between inf and sup (both included)"""
    return inf <= abs(err(e)) <= sup

def ratingsCountBetween(e, x='u', inf=0, sup=float('inf')):
    """return true if the number of rating for x ('u' or 'm') is between inf
    and sup (both included)"""
    xr = infos['ur'] if x == 'u' else infos['mr']
    x0 = 'u0' if x == 'u' else 'm0'
    return inf <= len(xr[e[x0]]) <= sup

# requirements that an estimation needs to fit
requirements = (lambda e: 
    errorBetween(e, inf=3.5)  and 
    ratingsCountBetween(e, x='m', sup=100))
# in example above, we look for all the estimations where abs(error) is more
# than 3.5 and where m0 has been rated less than 100 times.

# iterator with all estimations fitting the previously defined requirements
interstingEstimations = filter(requirements, infos['ests'])


# print details for estimations we are interested in
for e in interstingEstimations:
    details(e)
    print('-' * 52)
