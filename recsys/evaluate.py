"""
The :mod:`evaluate` module defines the :func:`evaluate` function.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import time
import os
import numpy as np

from . import accuracy
from .dump import dump


def evaluate(algo, data, measures=['rmse', 'mae'], with_dump=False,
             dump_dir=None, verbose=1):
    """Evaluate the performance of the algorithm on given data.

    Depending on the nature of the ``data`` parameter, it may or may not
    perform cross validation.

    Args:
        algo(:obj:`AlgoBase <recsys.prediction_algorithms.bases.AlgoBase>`):
            The algorithm to evaluate.
        data(:obj:`Dataset <recsys.dataset.Dataset>`): The dataset on which to
            evaluate the algorithm.
        measures(list of string): The performance measures to compute. Allowed
            names are function names as defined in the :mod:`accuracy
            <recsys.accuracy>` module. Default is ``['rmse', 'mae']``.
        with_dump(bool): If True, the predictions, the trainsets and the
            algorithm parameters will be dumped for later further analysis at
            each fold (see :ref:`User Guide <dumping>`).  The file names will
            be set as: ``'<date>-<algorithm name>-<fold number>'``.  Default is
            ``False``.
        dump_dir(str): The directory where to dump to files. Default is
            ``'~/.recsys/dumps/'``.
        verbose(int): Level of verbosity. If 0, nothing is printed. If 1
            (default), accuracy measures for each folds are printed, with a
            final summary. If 2, every prediction is printed.

    Returns:
        A dictionary containing measures as keys and lists as values. Each list
        contains one entry per fold.
    """

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

            if dump_dir is None:
                dump_dir = os.path.expanduser('~') + '/.recsys_data/dumps/'

            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)

            date = time.strftime('%y%m%d-%Hh%Mm%S', time.localtime())
            file_name = date + '-' + algo.__class__.__name__
            file_name += '-fold{0}'.format(fold_i)
            file_name = os.path.join(dump_dir, file_name)

            dump(file_name, predictions, trainset, algo)

    if verbose:
        print('-' * 20)
        for measure in measures:
            print('mean', measure.upper(),
                  ': {0:1.4f}'.format(np.mean(performances[measure])))

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
