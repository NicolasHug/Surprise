#!/usr/bin/python3

import random as rd
import numpy as np
import time
import argparse
import os

import stats
from prediction_algorithms import NormalPredictor
from prediction_algorithms import BaselineOnly
from prediction_algorithms import KNNBasic
from prediction_algorithms import KNNBaseline
from prediction_algorithms import KNNWithMeans
from prediction_algorithms import Parall
from prediction_algorithms import Pattern
from prediction_algorithms import CloneBruteforce
from prediction_algorithms import CloneMeanDiff
from prediction_algorithms import CloneKNNMeanDiff
from dataset import get_raw_ratings
from dataset import TrainingData


def main():

    parser = argparse.ArgumentParser(
        description='run a prediction algorithm for recommendation on given '
        'folds',
        epilog='example: main.py -algo KNNBasic -cv 3 -k 30 -sim cos '
        '--item_based')

    algo_choices = {
        'Normal': NormalPredictor,
        'BaselineOnly': BaselineOnly,
        'KNNBasic': KNNBasic,
        'KNNBaseline': KNNBaseline,
        'KNNWithMeans': KNNWithMeans,
        'Parall': Parall,
        'Pattern': Pattern,
        'CloneBruteforce': CloneBruteforce,
        'CloneMeanDiff': CloneMeanDiff,
        'CloneKNNMeanDiff': CloneKNNMeanDiff
    }
    parser.add_argument('-algo', type=str,
                        default='KNNBaseline',
                        choices=algo_choices,
                        help='the prediction algorithm to use. ' +
                        'Allowed values are ' + ', '.join(algo_choices.keys()) +
                        '. (default: KNNBaseline)',
                        metavar='<prediction algorithm>')

    sim_choices = ['cos', 'pearson', 'MSD', 'MSDClone']
    parser.add_argument('-sim', type=str,
                        default='MSD',
                        choices=sim_choices,
                        help='for algorithms using a similarity measure. ' +
                        'Allowed values are ' + ', '.join(sim_choices) + '.' +
                        ' (default: MSD)', metavar=' < sim measure >')

    method_choices = ['als', 'sgd']
    parser.add_argument('-method', type=str,
                        default='als',
                        choices=method_choices,
                        help='for algorithms using a baseline, the method ' +
                        'to compute it. Allowed values are ' +
                        ', '.join(method_choices) + '. (default: als)',
                        metavar='<method>')

    parser.add_argument('-k', type=int,
                        metavar='<number of neighbors>',
                        default=40,
                        help='the number of neighbors to use for k-NN ' +
                        'algorithms (default: 40)')

    parser.add_argument('-train_file', type=str,
                        metavar='<train file>',
                        default=None,
                        help='the file containing raw ratings for training. ' +
                        'The dataset argument needs to be set accordingly ' +
                        '(default: None)')

    parser.add_argument('-test_file', type=str,
                        metavar='<test file>',
                        default=None,
                        help='the file containing raw ratings for testing. ' +
                        'The dataset argument needs to be set accordingly. ' +
                        '(default: None)')

    dataset_choices = ['ml-100k', 'ml-1m', 'BX', 'jester']
    parser.add_argument('-dataset', metavar='<dataset>', type=str,
                        default='ml-100k',
                        choices=dataset_choices,
                        help='the dataset to use. Allowed values are ' +
                        ', '.join(dataset_choices) +
                        '( default: ml-100k -- MovieLens 100k)')

    parser.add_argument('-cv', type=int,
                        metavar="<number of folds>",
                        default=5,
                        help='the number of folds for cross validation. ' +
                        'Ignored if train_file and test_file are set. ' +
                        '(default: 5)')

    parser.add_argument('-seed', type=int,
                        metavar='<random seed>',
                        default=None,
                        help='the seed to use for RNG ' +
                        '(default: current system time)')

    parser.add_argument('--item_based', dest='item_based', action='store_const',
                        const=True,
                        default=False,
                        help='compute similarities on items rather than ' +
                        'on users')

    parser.add_argument('--with_dump', dest='with_dump', action='store_const',
                        const=True, default=False, help='tells to dump ' +
                        'results in a file (default: False)')

    parser.add_argument('--indiv_output', dest='indiv_output',
                        action='store_const',
                        const=True,
                        default=False,
                        help='to print individual prediction results ' +
                        '(default: False)')

    args = parser.parse_args()

    rd.seed(args.seed)

    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    def k_folds(seq, k):
        """inpired from scikit learn KFold method"""
        rd.shuffle(seq)
        start, stop = 0, 0
        for fold in range(k):
            start = stop
            stop += len(seq) // k
            if fold < len(seq) % k:
                stop += 1
            yield seq[:start] + seq[stop:], seq[start:stop]

    rmses = []  # list of rmse: one per fold

    def get_rmse(train_raw_ratings, test_raw_ratings):
        reader_train = reader_klass(train_raw_ratings)
        reader_test = reader_klass(test_raw_ratings)

        training_data = TrainingData(reader_train)

        train_start_time = time.process_time()
        training_time = time.process_time() - train_start_time

        algo = algo_choices[args.algo](training_data,
                                      item_based=args.item_based,
                                      method=args.method,
                                      sim_name=args.sim,
                                      k=args.k,
                                      with_dump=args.with_dump)

        print("computing predictions...")
        test_start_time = time.process_time()
        for u0, i0, r0, _ in reader_test.ratings:

            if args.indiv_output:
                print(u0, i0, r0)

            try:
                u0 = training_data.raw2inner_id_users[u0]
                i0 = training_data.raw2inner_id_items[i0]
            except KeyError:
                if args.indiv_output:
                    print("user or item wasn't used for training. Skipping")
                continue

            algo.predict(u0, i0, r0, args.indiv_output)

            if args.indiv_output:
                print('-' * 15)

        testingTime = time.process_time() - test_start_time

        if args.indiv_output:
            print('-' * 20)

        algo.infos['training_time'] = training_time
        algo.infos['testingTime'] = testingTime

        algo.dump_infos()
        print('-' * 20)
        print('Results:')
        return stats.compute_stats(algo.preds)

    if args.train_file and args.test_file:
        train_raw_ratings, reader_klass = get_raw_ratings(args.dataset,
                                                          args.train_file)
        test_raw_ratings, reader_klass = get_raw_ratings(args.dataset,
                                                         args.test_file)
        rmses.append(get_rmse(train_raw_ratings, test_raw_ratings))

    else:
        raw_ratings, reader_klass = get_raw_ratings(args.dataset)
        for fold_i, (training_set, test_set) in enumerate(k_folds(raw_ratings,
                                                                  args.cv)):
            print('-' * 19)
            print("-- fold numer {0} --".format(fold_i + 1))
            print('-' * 19)
            rmses.append(get_rmse(training_set, test_set))

    print(args)
    print("Mean RMSE: {0:1.4f}".format(np.mean(rmses)))

if __name__ == "__main__":
    main()
