"""
The :mod:`evaluate` module defines the :func:`evaluate` function and :class:`GridSearch` class
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import time
import os

import numpy as np
from six import iteritems
from six import itervalues
from itertools import product

from . import accuracy
from .dump import dump


def evaluate(algo, data, measures=['rmse', 'mae'], with_dump=False,
             dump_dir=None, verbose=1):
    """Evaluate the performance of the algorithm on given data.

    Depending on the nature of the ``data`` parameter, it may or may not
    perform cross validation.

    Args:
        algo(:obj:`AlgoBase \
            <surprise.prediction_algorithms.algo_base.AlgoBase>`):
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

        row_format = '{:<8}' * (n_folds + 2)
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


class GridSearch:
    # TODO: keep documentation up-to-date with code
    """Evaluate the performance of the algorithm on all the combinations of parameters given to it.

        Used to get study the effect of parameters on algorithms and extract best parameters.

        Depending on the nature of the ``data`` parameter, it may or may not
        perform cross validation.

        Parameters:
            algo_class(:obj:`AlgoBase \
                <surprise.prediction_algorithms.algo_base.AlgoBase>`):
                The algorithm to evaluate.
            param_grid (dict): The dictionary has algo_class parameters as keys \
                (string) and list of parameters as the desired values to try. \
                All combinations will be evaluated with desired algorithm
            measures(list of string): The performance measures to compute. Allowed
                names are function names as defined in the :mod:`accuracy
                <surprise.accuracy>` module. Default is ``['rmse', 'mae']``.
            verbose(boolean): Whether to print results when executing predictions.
                Default is `True`
        """
    def __init__(self, algo_class, param_grid, measures=['rmse', 'mae'], verbose=True):
        self.best_params_ = CaseInsensitiveDefaultDict(list)
        self.best_index_ = CaseInsensitiveDefaultDict(list)
        self.best_score_ = CaseInsensitiveDefaultDict(list)
        self.cv_results_ = {}
        self.algo_class = algo_class
        self.param_grid = param_grid
        self.measures = measures
        self.verbose = verbose
        self.param_combinations = [dict(zip(param_grid, v)) for v in product(*param_grid.values())]

    def evaluate(self, data):
        params = []
        scores = []
        combination_counter = 0
        for combination in self.param_combinations:
            params.append(combination)
            performances = CaseInsensitiveDefaultDict(list)
            if self.verbose:
                combination_counter += 1
                num_of_combinations = len(self.param_combinations)
                print ('start combination {} from {}: '.format(combination_counter,num_of_combinations))
                print ('params: ', combination)

            algo_instance = self.algo_class(**combination)
            for fold_i, (trainset, testset) in enumerate(data.folds()):
                if self.verbose:
                    print('-' * 12)
                    print('Fold ' + str(fold_i + 1))
                algo_instance.train(trainset)
                predictions = algo_instance.test(testset)
                for measure in self.measures:
                    f = getattr(accuracy, measure.lower())
                    performances[measure].append(f(predictions, verbose=self.verbose))

            mean_score = {}
            for measure in self.measures:
                mean_score[measure.upper()] = np.mean(performances[measure])

            if self.verbose:
                print('-' * 12)
                print('-' * 12)
                for measure in self.measures:
                    print('Mean {0:4s}: {1:1.4f}'.format(
                        measure.upper(), np.mean(performances[measure])))
                print('-' * 12)
                print('-' * 12)

            scores.append(mean_score)

        self.cv_results_ = {'params': params, 'scores': scores}

        for param in params:
            for key in param.keys():
                if key in self.cv_results_.keys():
                    self.cv_results_[key].append(param[key])
                else:
                    self.cv_results_[key] = [param[key]]
        for score in scores:
            for key in score.keys():
                if key in self.cv_results_.keys():
                    self.cv_results_[key].append(score[key])
                else:
                    self.cv_results_[key] = [score[key]]

        for measure in self.measures:
            #TODO: Check if it is okay to have hardcoded measures
            if measure.upper() == 'FCP':
                best_dict = max(self.cv_results_['scores'], key=lambda x: x[measure.upper()])
            else:
                best_dict = min(self.cv_results_['scores'], key=lambda x: x[measure.upper()])
            self.best_score_[measure] = best_dict[measure.upper()]
            self.best_index_[measure] = self.cv_results_['scores'].index(best_dict)
            self.best_params_[measure] = self.cv_results_['params'][self.best_index_[measure]]
            print (self.cv_results_)