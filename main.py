#!/usr/bin/python3
import random as rd
from collections import defaultdict
import numpy as np
import time
import sys
import argparse
from scipy.sparse import dok_matrix
from urllib.request import urlretrieve
import zipfile

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
        help='for algorithms using a similarity measure. Allowed values are '
        + ', '.join(simChoices) + '. (default: MSD)',
        metavar='<sim measure>')

methodChoices = ['als', 'sgd']
parser.add_argument('-method', type=str,
        default='als',
        choices=methodChoices,
        help='for algorithms using a baseline, the method to compute it. '
        'Allowed values are ' + ', '.join(simChoices) + '. (default: als)',
        metavar='<method>')

parser.add_argument('-k', type=int,
        metavar = '<number of neighbors>',
        default=40,
        help='the number of neighbors to use for k-NN algorithms (default: 40)')

parser.add_argument('-trainFile', type=str,
        metavar = '<train file>',
        default=None,
        help='the file containing raw ratings for training. dataset argument '
        'needs to be set accordingly (default: None)')

parser.add_argument('-testFile', type=str,
        metavar = '<test file>',
        default=None,
        help='the file containing raw ratings for testing. dataset argument '
        'needs to be set accordingly. (default: None)')

datasetChoices = ['ml-100k', 'ml-1m', 'BX']
parser.add_argument('-dataset', metavar='<dataset>', type=str,
        default='ml-100k',
        choices=datasetChoices,
        help='the dataset to use. Allowed values are ' +
        ', '.join(datasetChoices) + '( default: ml-100k -- MovieLens 100k)')

parser.add_argument('-cv', type=int,
        metavar = "<number of folds>",
        default=5,
        help='the number of folds for cross validation. Ignored if trainFile '
        'and testFile are set. (default: 5)')

parser.add_argument('-seed', type=int,
        metavar = '<random seed>',
        default=None,
        help='the seed to use for RNG (default: current system time)')

parser.add_argument('--itemBased', dest='itemBased', action='store_const',
const=True, default=False, help='compute similarities on items rather than on '
        'users')

parser.add_argument('--withDump', dest='withDump', action='store_const',
        const=True, default=False, help='tells to dump results in a file '
        '(default: False)')

parser.add_argument('--indivOutput', dest='indivOutput', action='store_const',
        const=True, default=False, help='to print individual prediction '
        'results (default: False)')

args = parser.parse_args()

rd.seed(args.seed)

if not os.path.exists('datasets'):
    os.makedirs('datasets')

def downloadDataset(dataset):
    answered = False
    while not answered:
        print('dataset ' + dataset + ' could not be found. Do you want to '
        'download it? [Y/n]')
        choice = input().lower()
        if choice in ['yes', 'y', '']:
            answered = True
        elif choice in ['no', 'n']:
            answered = True
            print("Ok then, I'm out")
            sys.exit()

    if dataset == 'ml-100k':
        url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    elif dataset == 'ml-1m':
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    elif dataset == 'BX':
        url = 'http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip'

    print('downloading...')
    urlretrieve(url, 'tmp.zip')
    print('done')

    zf = zipfile.ZipFile('tmp.zip', 'r')
    zf.extractall('datasets/' + dataset)
    os.remove('tmp.zip')

def getRawRatings(dataset, trainFile=None):
    if dataset == 'ml-100k':
        dataFile = trainFile or './datasets/ml-100k/ml-100k/u.data'
        ReaderClass = MovieLens100kReader
    elif dataset == 'ml-1m':
        dataFile = trainFile or './datasets/ml-1m/ml-1m/ratings.dat'
        ReaderClass = MovieLens1mReader
    elif dataset == 'BX':
        ReaderClass = BXReader
        dataFile = trainFile or './datasets/BX/BX-Book-Ratings.csv'

    try:
        f = open(dataFile, 'r')
    except FileNotFoundError:
        downloadDataset(dataset)
        f = open(dataFile, 'r')

    data = [line for line in f]

    return data, ReaderClass

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

def getRmse(trainRawRatings, testRawRatings):
    readerTrain = ReaderClass(trainRawRatings)
    readerTest = ReaderClass(testRawRatings)

    trainingData = TrainingData(readerTrain)

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

    algo.dumpInfos()
    print('-' * 20)
    print('-' * 20)
    return c.computeStats(algo.preds)

if args.trainFile and args.testFile:
    trainRawRatings, ReaderClass = getRawRatings(args.dataset, args.trainFile)
    testRawRatings, ReaderClass = getRawRatings(args.dataset, args.testFile)
    rmses.append(getRmse(trainRawRatings, testRawRatings))

else:
    rawRatings, ReaderClass = getRawRatings(args.dataset)
    for foldNumber, (trainSet, testSet) in enumerate(kFolds(rawRatings, args.cv)):
        print("-- fold numer {0} --".format(foldNumber + 1))
        rmses.append(getRmse(trainSet, testSet))


print(args)
print("RMSE: {0:1.4f}".format(np.mean(rmses)))
