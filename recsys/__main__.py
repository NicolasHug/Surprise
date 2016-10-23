#!/usr/bin/env python

import random as rd
import numpy as np
import time
import argparse
import os

from . import accuracy
from recsys.prediction_algorithms import NormalPredictor
from recsys.prediction_algorithms import BaselineOnly
from recsys.prediction_algorithms import KNNBasic
from recsys.prediction_algorithms import KNNBaseline
from recsys.prediction_algorithms import KNNWithMeans
from recsys.dataset import Dataset
from recsys.evaluate import evaluate


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
        'KNNWithMeans': KNNWithMeans
    }
    parser.add_argument('-algo', type=str,
                        default='KNNBaseline',
                        choices=algo_choices,
                        help='the prediction algorithm to use. ' +
                        'Allowed values are ' + ', '.join(algo_choices.keys()) +
                        '. (default: KNNBaseline)',
                        metavar='<prediction algorithm>')

    sim_choices = ['cos', 'pearson', 'MSD', 'pearson_baseline']
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

    parser.add_argument('-shrinkage', type=int,
                        metavar='<shrinkge value>',
                        default=100,
                        help='the shrinkage value to use for pearson ' +
                        'similarity (default: 100)')

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
                        help='the number of folds for cross-validation. ' +
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
    np.random.seed(args.seed)

    algo = algo_choices[args.algo](user_based=not args.item_based,
                                   method=args.method,
                                   sim_name=args.sim,
                                   k=args.k,
                                   shrinkage=args.shrinkage)

    data = Dataset.load_builtin(args.dataset)
    data.split(n_folds=args.cv)
    evaluate(algo, data, args.with_dump)

if __name__ == "__main__":
    main()
