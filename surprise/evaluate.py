"""The :mod:`evaluate` module defines the :func:`evaluate` function and
:class:`GridSearch` class """

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


class GridSearch:
    """The :class:`GridSearch` class, used to evaluate the performance of an
    algorithm on various combinations of parameters, and extract the best
    combination. It is analogous to `GridSearchCV
    <http://scikit-learn.org/stable/modules/generated/sklearn.
    model_selection.GridSearchCV.html>`_ from scikit-learn.

    See :ref:`User Guide <tuning_algorithm_parameters>` for usage.

    Args:
        algo_class(:obj:`AlgoBase \
            <surprise.prediction_algorithms.algo_base.AlgoBase>`):
            A class object of of the algorithm to evaluate.
        param_grid (dict):
            The dictionary has algo_class parameters as keys (string) and list
            of parameters as the desired values to try.  All combinations will
            be evaluated with desired algorithm.
        measures(list of string):
            The performance measures to compute. Allowed names are function
            names as defined in the :mod:`accuracy <surprise.accuracy>` module.
            Default is ``['rmse', 'mae']``.
        verbose(int):
            Level of verbosity. If ``0``, nothing is printed. If ``1``,
            accuracy measures for each parameters combination are printed, with
            combination values. If ``2``, folds accuracy values are also
            printed.  Default is ``1``.

    Attributes:
        cv_results (dict of arrays):
            A dict that contains all parameters and accuracy information for
            each combination. Can  be imported into a pandas `DataFrame`.
        best_estimator (dict of AlgoBase):
            Using an accuracy measure as key, get the estimator that gave the
            best accuracy results for the chosen measure.
        best_score (dict of floats):
            Using an accuracy measure as key, get the best score achieved for
            that measure.
        best_params (dict of dicts):
            Using an accuracy measure as key, get the parameters combination
            that gave the best accuracy results for the chosen measure.
        best_index  (dict of ints):
            Using an accuracy measure as key, get the index that can be used
            with `cv_results_` that achieved the highest accuracy for that
            measure.
        """

    def __init__(self, algo_class, param_grid, measures=['rmse', 'mae'],
                 verbose=1):
        self.best_params = CaseInsensitiveDefaultDict(list)
        self.best_index = CaseInsensitiveDefaultDict(list)
        self.best_score = CaseInsensitiveDefaultDict(list)
        self.best_estimator = CaseInsensitiveDefaultDict(list)
        self.cv_results = defaultdict(list)
        self.algo_class = algo_class
        self.param_grid = param_grid
        self.measures = [measure.upper() for measure in measures]
        self.verbose = verbose
        self.param_combinations = [dict(zip(param_grid, v)) for v in
                                   product(*param_grid.values())]

    def evaluate(self, data):
        """Runs the grid search on dataset.

        Class instance attributes can be accessed after the evaluate is done.

        Args:
            data (:obj:`Dataset <surprise.dataset.Dataset>`): The dataset on
                which to evaluate the algorithm.
        """

        num_of_combinations = len(self.param_combinations)
        params = []
        scores = []

        # evaluate each combination of parameters using the evaluate method
        for combination_index, combination in enumerate(
                self.param_combinations):
            params.append(combination)

            if self.verbose >= 1:
                print('-' * 12)
                print('Parameters combination {} of {}'.
                      format(combination_index + 1, num_of_combinations))
                print('params: ', combination)

            # the algorithm to use along with the combination parameters
            algo_instance = self.algo_class(**combination)
            evaluate_results = evaluate(algo_instance, data,
                                        measures=self.measures,
                                        verbose=(self.verbose == 2))

            # measures as keys and folds average as values
            mean_score = {}
            for measure in self.measures:
                mean_score[measure] = np.mean(evaluate_results[measure])
            scores.append(mean_score)

            if self.verbose == 1:
                print('-' * 12)
                for measure in self.measures:
                    print('Mean {0:4s}: {1:1.4f}'.format(
                        measure, mean_score[measure]))
                print('-' * 12)

        # Add all scores and parameters lists to dict
        self.cv_results['params'] = params
        self.cv_results['scores'] = scores

        # Add accuracy measures and algorithm parameters as keys to dict
        for param, score in zip(params, scores):
            for param_key, score_key in zip(param.keys(), score.keys()):
                self.cv_results[param_key].append(param[param_key])
                self.cv_results[score_key].append(score[score_key])

        # Get the best results
        for measure in self.measures:
            if measure == 'FCP':
                best_dict = max(self.cv_results['scores'],
                                key=lambda x: x[measure])
            else:
                best_dict = min(self.cv_results['scores'],
                                key=lambda x: x[measure])
            self.best_score[measure] = best_dict[measure]
            self.best_index[measure] = self.cv_results['scores'].index(
                best_dict)
            self.best_params[measure] = self.cv_results['params'][
                self.best_index[measure]]
            self.best_estimator[measure] = self.algo_class(
                **self.best_params[measure])


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


def print_perf(performances):

    # retrieve number of folds. Kind of ugly...
    n_folds = [len(values) for values in itervalues(performances)][0]

    row_format = '{:<8}' * (n_folds + 2)
    s = row_format.format(
        '',
        *['Fold {0}'.format(i + 1) for i in range(n_folds)] + ['Mean'])
    s += '\n'
    s += '\n'.join(row_format.format(
        key.upper(),
        *['{:1.4f}'.format(v) for v in vals] +
        ['{:1.4f}'.format(np.mean(vals))])
        for (key, vals) in iteritems(performances))

    print(s)
