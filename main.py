#!/usr/bin/python3
import random as rd
from collections import defaultdict
import numpy as np
import time
import sys
import argparse

import common as c
from algoAnalogy import *
from algoKorbenAndCo import *
from algoClone import *

parser = argparse.ArgumentParser(
        description='run a prediction algorithm for recommendation on given '
        'folds',
        epilog='example: main.py -algo AlgoBasicCollaborative -k 30 -sim cos '
        '--movieBased 1 3')

algoChoices = {
        'AlgoRandom'                   : AlgoRandom,
        'AlgoBaselineOnly'             : AlgoBaselineOnly,
        'AlgoBasicCollaborative'       : AlgoBasicCollaborative,
        'AlgoNeighborhoodWithBaseline' : AlgoNeighborhoodWithBaseline,
        'AlgoParall'                   : AlgoParall,
        'AlgoPattern'                  : AlgoPattern,
        'AlgoKNNBelkor'                : AlgoKNNBelkor,
        'AlgoFactors'                  : AlgoFactors,
        'AlgoCloneBruteforce'          : AlgoCloneBruteforce,
        'AlgoCloneMeanDiff'            : AlgoCloneMeanDiff,
        'AlgoCloneKNNMeanDiff'         : AlgoCloneKNNMeanDiff
}
parser.add_argument('-algo', type=str,
        default='AlgoNeighborhoodWithBaseline',
        choices=algoChoices,
        help='The prediction algorithm to use. Allowed values are '
        + ', '.join(algoChoices.keys()) + '. (default: '
        'AlgoNeighborhoodWithBaseline)',
        metavar='prediction_algo'
        )

simChoices = ['cos', 'pearson', 'MSD', 'MSDClone']
parser.add_argument('-sim', type=str,
        default='MSD',
        choices=simChoices,
        help='The similarity measure to use. Allowed values are '
        + ', '.join(simChoices) + '. (default: MSD)',
        metavar='sim_measure')

methodChoices = ['als', 'sgd']
parser.add_argument('-method', type=str,
        default='als',
        choices=methodChoices,
        help='The method to compute user and item biases. Allowed values are '
        + ', '.join(simChoices) + '. (default: als)',
        metavar='method')

parser.add_argument('-k', type=int,
        default=40,
        help='The number of neighbors to use (default: 40)')

parser.add_argument('folds', metavar='fold', type=int, nargs='*',
        default=[1, 2, 3, 4, 5],
        help='The fold numbers on which to make predictions (default: 1 2 3 4 '
        '5)')

parser.add_argument('--movieBased', dest='movieBased', action='store_const',
const=True, default=False, help='compute similarities on movies (default: user'
        ' based)')

parser.add_argument('--withDump', dest='withDump', action='store_const',
        const=True, default=False, help='tells to dump results in a file '
        '(default: False)')

parser.add_argument('--indivOutput', dest='indivOutput', action='store_const',
        const=True, default=False, help='to print individual prediction '
        'results (default: False)')

args = parser.parse_args()

rmses = [] # list of rmse: one per fold
for fold in args.folds:

    print("-- fold numer %d --" % fold)

    base = open('../ml-100k/u%s.base' % fold , 'r')
    test = open('../ml-100k/u%s.test' % fold, 'r')

    rm = np.zeros((c.lastMi + 1, c.lastUi + 1), dtype='int') # the rating matrix
    ur = defaultdict(list)  # dict of users containing list of (m, rat(u, m))
    mr = defaultdict(list)  # dict of movies containing list of (u, rat(u, m))

    # read training file
    for line in base:
        ui, mi, r, _ = line.split()
        ui = int(ui); mi = int(mi); r = int(r)
        rm[mi, ui] = r
        ur[ui].append((mi, r))
        mr[mi].append((ui, r))

    trainStartTime = time.process_time()
    trainingTime = time.process_time() - trainStartTime

    algo = algoChoices[args.algo](rm, ur, mr,
            movieBased=args.movieBased,
            method=args.method,
            sim=args.sim,
            k=args.k)

    algo.infos['params']['fold'] = fold

    rd.seed(0)
    testSet = []
    for line in test:
        testSet.append(line.split())

    smallTestSet = [rd.choice(testSet) for i in range(100)]

    print("computing predictions...")
    testTimeStart = time.process_time()
    for u0, m0, r0, _ in testSet:
    #for u0, m0, r0, _ in smallTestSet:

        u0 = int(u0); m0 = int(m0); r0 = int(r0)

        if args.indivOutput:
            print(u0, m0, r0)

        algo.estimate(u0, m0)
        algo.cut_estimate(1, 5)
        algo.updatePreds(u0, m0, r0, args.indivOutput)

        if args.indivOutput:
            print('-' * 20)

    testingTime = time.process_time() - testTimeStart

    if args.indivOutput:
        print('-' * 20)

    algo.infos['trainingTime'] = trainingTime
    algo.infos['testingTime'] = testingTime

    rmses.append(c.computeStats(algo.infos['preds']))
    algo.dumpInfos()
    print('-' * 20)
    print('-' * 20)

print("mean RMSE of", args.algo, "on folds", args.folds,
        ": {0:1.4f}".format(np.mean(rmses)))
