"""
The :mod:`evaluate` module defines the :func:`evaluate` function.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import time
import os

import numpy as np
from .six import iteritems
from .six import itervalues

from . import accuracy
from .dump import dump


def evaluate(algo, data, measures=['rmse', 'mae'], with_dump=False,
             dump_dir=None, verbose=1):
    """Evaluate the performance of the algorithm on given data.

    Depending on the nature of the ``data`` parameter, it may or may not
    perform cross validation.

    Args:
        algo(:obj:`AlgoBase <surprise.prediction_algorithms.bases.AlgoBase>`):
            The algorithm to evaluate.
        data(:obj:`Dataset <surprise.dataset.Dataset>`): The dataset on which
            to evaluate the algorithm.
        measures(list of string): The performance measures to compute. Allowed
            names are function names as defined in the :mod:`accuracy
            <surprise.accuracy>` module. Default is ``['rmse', 'mae']``.
        with_dump(bool): If True, the predictions, the trainsets and the
            algorithm parameters will be dumped for later further analysis at
            each fold (see :ref:`User Guide <dumping>`).  The file names will
            be set as: ``'<date>-<algorithm name>-<fold number>'``.  Default is
            ``False``.
        dump_dir(str): The directory where to dump to files. Default is
            ``'~/.surprise_data/dumps/'``.
        verbose(int): Level of verbosity. If 0, nothing is printed. If 1
            (default), accuracy measures for each folds are printed, with a
            final summary. If 2, every prediction is printed.

    Returns:
        A dictionary containing measures as keys and lists as values. Each list
        contains one entry per fold.
    """

    performances = CaseInsensitiveDefaultDict(list)
    print('Evaluating {0} of algorithm {1}.'.format(
          ', '.join((m.upper() for m in measures)),
          algo.__class__.__name__))
    print()

    for fold_i, (trainset, testset) in enumerate(data.folds()):

        if verbose:
            print('-' * 12)
            print('Fold ' + str(fold_i + 1))

        # train and test algorithm. Keep all rating predictions in a list
        algo.train(trainset)
        predictions = algo.test(testset, verbose=(verbose == 2))

        # compute needed performance statistics
        for measure in measures:
            f = getattr(accuracy, measure.lower())
            performances[measure].append(f(predictions, verbose=verbose))

        if with_dump:

            if dump_dir is None:
                dump_dir = os.path.expanduser('~') + '/.surprise_data/dumps/'

            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)

            date = time.strftime('%y%m%d-%Hh%Mm%S', time.localtime())
            file_name = date + '-' + algo.__class__.__name__
            file_name += '-fold{0}'.format(fold_i + 1)
            file_name = os.path.join(dump_dir, file_name)

            dump(file_name, predictions, trainset, algo)

    if verbose:
        print('-' * 12)
        print('-' * 12)
        for measure in measures:
            print('Mean {0:4s}: {1:1.4f}'.format(
                  measure.upper(), np.mean(performances[measure])))
        print('-' * 12)
        print('-' * 12)


    return performances


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

    def __str__(self):

        # retrieve number of folds. Kind of ugly...
        n_folds = [len(values) for values in itervalues(self)][0]

        row_format ='{:<8}' * (n_folds + 2)
        s = row_format.format(
            '',
            *['Fold {0}'.format(i + 1) for i in range(n_folds)] + ['Mean'])
        s += '\n'
        s += '\n'.join(row_format.format(
            key.upper(),
            *['{:1.4f}'.format(v) for v in vals] +
            ['{:1.4f}'.format(np.mean(vals))])
            for (key, vals) in iteritems(self))

        return s
