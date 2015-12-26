#!/usr/bin/python3
import random as rd
from collections import defaultdict
import numpy as np
import time
import sys
import argparse
from scipy.sparse import dok_matrix

import common as c
from algoAnalogy import *
from algoKorenAndCo import *
from algoClone import *
from dataset import *

parser = argparse.ArgumentParser(
        description='run a prediction algorithm for recommendation on given '
        'folds',
        epilog='example: main.py -algo AlgoKNNBasic -cv 3 -k 30 -sim cos '
        '--itemBased')

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
        help='the prediction algorithm to use. Allowed values are '
        + ', '.join(algoChoices.keys()) + '. (default: '
        'AlgoKNNBaseline)',
        metavar='<prediction algorithm>'
        )

simChoices = ['cos', 'pearson', 'MSD', 'MSDClone']
parser.add_argument('-sim', type=str,
        default='MSD',
        choices=simChoices,
        help='the similarity measure to use. Allowed values are '
        + ', '.join(simChoices) + '. (default: MSD)',
        metavar='<sim measure>')

methodChoices = ['als', 'sgd']
parser.add_argument('-method', type=str,
        default='als',
        choices=methodChoices,
        help='the method to compute user and item biases. Allowed values are '
        + ', '.join(simChoices) + '. (default: als)',
        metavar='<method>')

parser.add_argument('-k', type=int,
        metavar = '<number of neighbors>',
        default=40,
        help='the number of neighbors to use for k-NN algorithms (default: 40)')

parser.add_argument('-dataset', metavar='<dataset>', type=str,
        default='ml-100k',
        help='the dataset to use (default: ml-100k: MovieLens 100k)')

parser.add_argument('-cv', type=int,
        metavar = "<number of folds>",
        default=5,
        help='the number of folds for cross validation')

parser.add_argument('-seed', type=int,
        metavar = '<random seed>',
        default=None,
        help='the seed to use for RNG (default: current system time)')

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

rd.seed(args.seed)

if args.dataset == 'ml-100k':
    f = open('./datasets/ml-100k/u.data', 'r')
    data = [line for line in f]
    ReaderClass = MovieLens100kReader
elif args.dataset == 'ml-1m':
    f = open('./datasets/ml-1m/ratings.dat', 'r')
    data = [line for line in f]
    ReaderClass = MovieLens1mReader
elif args.dataset == 'BX':
    f = open('./datasets/BX/BX-Book-Ratings.csv', 'r')
    for line in f:
        print(line)
    data = [line for line in f][1:] # skip first line
    ReaderClass = BXReader

def kFolds(seq, k):
    """inpired from scikit learn KFold method"""
    rd.shuffle(seq)
    start, stop = 0, 0
    for fold in range(k):
        start = stop
        stop += len(seq) // k
        if fold < len(seq) % k:
            stop += 1
        yield seq[:start] + seq[stop:], seq[start:stop]

rmses = [] # list of rmse: one per fold
for foldNumber, (trainSet, testSet) in enumerate(kFolds(data, args.cv)):
    readerTrain = ReaderClass(trainSet)
    readerTest = ReaderClass(testSet)

    trainingData = TrainingData(readerTrain)

    print("-- fold numer {0} --".format(foldNumber + 1))

    trainStartTime = time.process_time()
    trainingTime = time.process_time() - trainStartTime

    algo = algoChoices[args.algo](trainingData,
            itemBased=args.itemBased,
            method=args.method,
            sim=args.sim,
            k=args.k)

    print("computing predictions...")
    testTimeStart = time.process_time()
    for u0, i0, r0, _ in readerTest.ratings:

        if args.indivOutput:
            print(u0, i0, r0)

        try:
            u0 = trainingData.rawToInnerIdUsers[u0]
            i0 = trainingData.rawToInnerIdItems[i0]
        except KeyError:
            if args.indivOutput:
                print("user or item wasn't used for training. Skipping")
            continue

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
