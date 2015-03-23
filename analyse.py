#!/usr/bin/python3

import pickle
import sys
import os

import common as c
from analyseTools import *

def analyseDumpFile(dumpFile):
    """print some info about an individual dump file"""
    print('File:', dumpFile)

    infos = pickle.load(open(dumpFile, 'rb'))

    # print name and specific parameters
    print('Algo name:', infos['name'])
    print('Tests count:', len(infos['preds']))
    for k, v in infos['params'].items():
        print(k + ':', v)


    # requirements that a prediction needs to fit
    requirements = (lambda p: p['u0'] == 334 and p['m0'] == 160)

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
    print('-' * 52)
    for p in interestingPreds:
        details(p, infos)
        print('-' * 52)

    print('-' * 52)
    # print RMSE & Co for these predictions
    c.printStats(interestingPreds)

        
    # print proportion of absolute errors
    print('-' * 10, "\n" + "Proportions of absolute errors among int. preds:")
    printHist(interestingPreds, key='err')
    print('-' * 10, "\n" + "Proportions of absolute errors among all preds:")
    printHist(infos['preds'], key='err')

    # print propotion of estimation values
    print('-' * 52, "\n" + "Proportions of rounded ests values in int. preds:")
    printHist(interestingPreds, key='est')
    print('-' * 10, "\n" + "Proportions of rounded ests values in all preds:")
    printHist(infos['preds'], key='est')

    # print propotion of rounded estimation values
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


def compareDumps(dumpFileA, dumpFileB):
    """compare two algorithms from their dumpFile"""

    print('File A:', dumpFileA)
    print('File B:', dumpFileB)

    infosA = pickle.load(open(dumpFileA, 'rb'))
    infosB = pickle.load(open(dumpFileB, 'rb'))

    for infos in (infosA, infosB):
        print('-' * 50)
        # print name and specific parameters
        print('Algo name:', infos['name'])
        print('Tests count:', len(infos['preds']))
        for k, v in infos['params'].items():
            print(k + ':', v)

    predsA = infosA['preds']
    predsB = infosB['preds']


    # show details of pred from A and B where pred for A is bad (err >= 3) or
    # good (err <= 1)
    """
    badPredsA = [(i, p) for (i, p) in enumerate(predsA) if errorBetween(p,
        inf=3)]
    goodPredsA = [(i, p) for (i, p) in enumerate(predsA) if errorBetween(p,
        sup=1)]

    print('-' * 52)
    print('BAD PREDS FOR A:')
    for (i, p) in badPredsA:
        print('Algo A')
        details(p, infosA)
        print('-' * 10)
        print('Algo B')
        details(predsB[i], infosB)
        print('-' * 52)
    print('-' * 52)
    print('GOOD PREDS FOR A:')
    for (i, p) in goodPredsA:
        print('Algo A')
        details(p, infosA)
        print('-' * 10)
        print('Algo B')
        details(predsB[i], infosB)
        print('-' * 52)
    """


    # show details of pred from A and B where A is better than B 
    # (err(A) <= err(B))
    print('-' * 50)
    aIsBetter = [i for (i, p) in enumerate(predsA) if abs(err(p)) <= abs(err(predsB[i]))]
    for i in aIsBetter:
        print('Algo A')
        details(predsA[i], infosA)
        print('-' * 10)
        print('Algo B')
        details(predsB[i], infosB)
        print('-' * 52)

    print('Algo A is better on {0} predictions, where:'.format(len(aIsBetter)))
    print('\tmean of abs errors for A: {0:1.2f}'.format(
        np.mean([abs(err(predsA[i])) for i in aIsBetter])))
    print('\tmean of abs errors for B: {0:1.2f}'.format(
        np.mean([abs(err(predsB[i])) for i in aIsBetter])))



argc = len(sys.argv)
if argc < 3:

    # by default, open the last created file in the dumps directory
    if len(sys.argv) < 2:
        for dirname, dirnames, filenames in os.walk('./dumps'):
            dumpFile = max([f for f in filenames], 
                key=lambda x:os.path.getctime(os.path.join(dirname, x)))
            dumpFile = os.path.join(dirname, dumpFile)
    # if file name is passed as argument, chose it instead
    elif len(sys.argv) < 3:
        dumpFile = sys.argv[1]

    analyseDumpFile(dumpFile)

else :
    dumpFileA = sys.argv[1]
    dumpFileB = sys.argv[2]
    compareDumps(dumpFileA, dumpFileB)
