"""
The :mod:`evaluate` module defines the :func:`evaluate` function.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import pickle
import time
import os
import numpy as np

from . import accuracy


def evaluate(algo, data, measures=['rmse', 'mae', 'fcp'], with_dump=False,
             verbose=1):
    """Evaluate the performance of the algorithm on given data.

    Depending on the nature of the ``data`` parameter, it may or may not
    perform cross validation.

    Args:
        algo(:obj:`AlgoBase <recsys.prediction_algorithms.bases.AlgoBase>`):
            The algorithm to evaluate.
        data(:obj:`Dataset <recsys.dataset.Dataset>`): The dataset on which to
            evaluate the algorithm.
        with_dump(bool): If True, the algorithm parameters and every prediction
            prediction will be dumped (using `Pickle
            <https://docs.python.org/3/library/pickle.html>`_) for potential
            further analysis. Default is ``False``.
        measures(list of string): The performance measures to compute. Allowed
            names are function names as defined in the :mod:`accuracy
            <recsys.accuracy>` module. Default is ``['rmse', 'mae', 'fcp']``.
        verbose(int): Level of verbosity. If 0, nothing is printed. If 1
            (default), accuracy measures for each folds are printed, with a
            final summary. If 2, every prediction is printed.

    Returns:
        A dictionary containing measures as keys and lists as values. Each list
        contains one entry per fold.
    """

    dump = {}
    performances = CaseInsensitiveDefaultDict(list)

    for fold_i, (trainset, testset) in enumerate(data.folds()):

        if verbose:
            print('-' * 20)
            print('fold ' + str(fold_i))

        # train and test algorithm. Keep all rating predictions in a list
        algo.train(trainset)
        predictions = algo.test(testset, verbose=(verbose == 2))

        # compute needed performance statistics
        for measure in measures:
            f = getattr(accuracy, measure.lower())
            performances[measure].append(f(predictions, verbose=verbose))

        if with_dump:
            fold_dump = dict(trainset=trainset, predictions=predictions)
            dump['fold_' + str(fold_i)] = fold_dump

    if verbose:
        print('-' * 20)
        for measure in measures:
            print('mean', measure.upper(),
                  ': {0:1.4f}'.format(np.mean(performances[measure])))

    if with_dump:
        dump['user_based'] = algo.user_based
        dump['algo'] = algo.__class__.__name__
        dump_evaluation(dump)

    return performances


def dump_evaluation(dump):

    dump_dir = os.path.expanduser('~') + '/.recsys_data/dumps/'

    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    date = time.strftime('%y%m%d-%Hh%Mm%S', time.localtime())
    name = (dump_dir + date + '-' + dump['algo'])

    pickle.dump(dump, open(name, 'wb'))
    print('File has been dumped to', dump_dir)


class CaseInsensitiveDefaultDict(defaultdict):
    """From here:
        http://stackoverflow.com/questions/2082152/case-insensitive-dictionary.

        As pointed out in the comments, this only covers a few cases and we
        should override a lot of other methods, but oh well...

        Used for the returned dict, so that users can use perf['RMSE'] or
        perf['rmse'] indifferently.
    """
    def __setitem__(self, key, value):
        super(CaseInsensitiveDefaultDict, self).__setitem__(key.lower(), value)

    def __getitem__(self, key):
        return super(CaseInsensitiveDefaultDict, self).__getitem__(key.lower())
