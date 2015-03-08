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

def err(est):
    """return the error between the expected rating and the estimated one"""
    return est['est'] - est['r0']

def details(est):
    """print details on an estimation"""
    # ids, true rating, etimation and error
    print("u0: {0:<3d}    m0: {1:<4d}   r0: {2}   est: {3:1.2f}"
        "   err: {4:-2.2f}".format(est['u0'], est['m0'], est['r0'], est['est'],
        err(est)))
    # u0 ratings mean and count
    print("u0 ratings:")
    print("\tcount: {0:d}".format(len(infos['ur'][est['u0']])))
    print("\tmean : {0:1.4f}".format(np.mean(infos['ur'][est['u0']], 0)[1]))
    # m0 ratings mean and count
    print("m0 ratings:")
    print("\tcount: {0:d}".format(len(infos['mr'][est['m0']])))
    if infos['mr'][est['m0']]:
        print("\tmean : {0:1.4f}".format(np.mean(infos['mr'][est['m0']], 0)[1]))
    else:
        print("\tmean : no f***ing way!")
        

def filterByError(ests, inf=0., sup=4.):
    """return an iterator with all the estimations where error is more or equl
    to inf and less or equal to sup"""
    return filter(lambda x: abs(err(x)) >= inf and abs(err(x)) <= sup, ests)

# detail of predictions
for est in filterByError(infos['ests'], inf=2):
    details(est)
    print('-' * 52)
