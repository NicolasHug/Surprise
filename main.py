#!/usr/bin/python3
import random as rd
from collections import defaultdict
import numpy as np
import time
import sys
import argparse

import common as c
from algoAnalogy import *
from algoKorenAndCo import *
from algoClone import *

parser = argparse.ArgumentParser(
        description='run a prediction algorithm for recommendation on given '
        'folds',
        epilog='example: main.py -algo AlgoKNNBasic -k 30 -sim cos '
        '--itemBased 1 3')

algoChoices = {
        'AlgoRandom'           : AlgoRandom,
        'AlgoBaselineOnly'     : AlgoBaselineOnly,
        'AlgoKNNBasic'         : AlgoKNNBasic,
        'AlgoKNNBaseline'      : AlgoKNNBaseline,
        'AlgoKNNWithMeans'     : AlgoKNNWithMeans,
        'AlgoParall'           : AlgoParall,
        'AlgoPattern'          : AlgoPattern,
        'AlgoKNNBelkor'        : AlgoKNNBelkor,
        'AlgoFactors'          : AlgoFactors,
        'AlgoCloneBruteforce'  : AlgoCloneBruteforce,
        'AlgoCloneMeanDiff'    : AlgoCloneMeanDiff,
        'AlgoCloneKNNMeanDiff' : AlgoCloneKNNMeanDiff
}
parser.add_argument('-algo', type=str,
        default='AlgoKNNBaseline',
        choices=algoChoices,
        help='The prediction algorithm to use. Allowed values are '
        + ', '.join(algoChoices.keys()) + '. (default: '
        'AlgoKNNBaseline)',
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

parser.add_argument('--itemBased', dest='itemBased', action='store_const',
const=True, default=False, help='compute similarities on items (default: user'
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

    base = open('./datasets/ml-100k/u%s.base' % fold , 'r')
    test = open('./datasets/ml-100k/u%s.test' % fold, 'r')

    rm = np.zeros((c.lastIi + 1, c.lastUi + 1), dtype='int')
    ur = defaultdict(list)
    ir = defaultdict(list)

    # read training file
    for line in base:
        ui, ii, r, _ = line.split()
        ui = int(ui); ii = int(ii); r = int(r)
        rm[ii, ui] = r
        ur[ui].append((ii, r))
        ir[ii].append((ui, r))

    trainStartTime = time.process_time()
    trainingTime = time.process_time() - trainStartTime

    algo = algoChoices[args.algo](rm, ur, ir,
            itemBased=args.itemBased,
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
    for u0, i0, r0, _ in testSet:
    #for u0, i0, r0, _ in smallTestSet:

        u0 = int(u0); i0 = int(i0); r0 = int(r0)

        if args.indivOutput:
            print(u0, i0, r0)

        algo.predict(u0, i0, r0, args.indivOutput)

        if args.indivOutput:
            print('-' * 15)

    testingTime = time.process_time() - testTimeStart

    if args.indivOutput:
        print('-' * 20)

    algo.infos['trainingTime'] = trainingTime
    algo.infos['testingTime'] = testingTime

    rmses.append(c.computeStats(algo.preds))
    algo.dumpInfos()
    print('-' * 20)
    print('-' * 20)

print(args)
print("RMSE: {0:1.4f}".format(np.mean(rmses)))
