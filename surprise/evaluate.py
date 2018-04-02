"""The :mod:`evaluate <surprise.evaluate>` module defines the :func:`evaluate`
function and :class:`GridSearch` class """

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict
import time
import os
from itertools import product
import random
import warnings

import numpy as np
from six import iteritems
from six import itervalues
from joblib import Parallel
from joblib import delayed

from .builtin_datasets import get_dataset_dir
from . import accuracy
from .dump import dump


def evaluate(algo, data, measures=['rmse', 'mae'], with_dump=False,
             dump_dir=None, verbose=1):
    """
    .. warning::
        Deprecated since version 1.05.  Use :func:`cross_validate
        <surprise.model_selection.validation.cross_validate>` instead. This
        function will be removed in later versions.

    Evaluate the performance of the algorithm on given data.

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
        with_dump(bool): If True, the predictions and the algorithm will be
            dumped for later further analysis at each fold (see :ref:`FAQ
            <serialize_an_algorithm>`). The file names will be set as:
            ``'<date>-<algorithm name>-<fold number>'``.  Default is ``False``.
        dump_dir(str): The directory where to dump to files. Default is
            ``'~/.surprise_data/dumps/'``, or the folder specified by the
            ``'SURPRISE_DATA_FOLDER'`` environment variable (see :ref:`FAQ
            <data_folder>`).
        verbose(int): Level of verbosity. If 0, nothing is printed. If 1
            (default), accuracy measures for each folds are printed, with a
            final summary. If 2, every prediction is printed.

    Returns:
        A dictionary containing measures as keys and lists as values. Each list
        contains one entry per fold.
    """

    warnings.warn('The evaluate() method is deprecated. Please use '
                  'model_selection.cross_validate() instead.', UserWarning)

    performances = CaseInsensitiveDefaultDict(list)

    if verbose:
        print('Evaluating {0} of algorithm {1}.'.format(
              ', '.join((m.upper() for m in measures)),
              algo.__class__.__name__))
        print()

    for fold_i, (trainset, testset) in enumerate(data.folds()):

        if verbose:
            print('-' * 12)
            print('Fold ' + str(fold_i + 1))

        # train and test algorithm. Keep all rating predictions in a list
        algo.fit(trainset)
        predictions = algo.test(testset, verbose=(verbose == 2))

        # compute needed performance statistics
        for measure in measures:
            f = getattr(accuracy, measure.lower())
            performances[measure].append(f(predictions, verbose=verbose))

        if with_dump:

            if dump_dir is None:
                dump_dir = os.path.join(get_dataset_dir(), 'dumps/')

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
    """
    .. warning::
        Deprecated since version 1.05. Use :func:`GridSearchCV
        <surprise.model_selection.search.GridSearchCV>` instead. This
        class will be removed in later versions.

    The :class:`GridSearch` class, used to evaluate the performance of an
    algorithm on various combinations of parameters, and extract the best
    combination. It is analogous to `GridSearchCV
    <http://scikit-learn.org/stable/modules/generated/sklearn.
    model_selection.GridSearchCV.html>`_ from scikit-learn.

    See :ref:`User Guide <tuning_algorithm_parameters>` for usage.

    Args:
        algo_class(:obj:`AlgoBase \
            <surprise.prediction_algorithms.algo_base.AlgoBase>`): The class
            object of the algorithm to evaluate.
        param_grid(dict): Dictionary with algorithm parameters as keys and
            list of values as keys. All combinations will be evaluated with
            desired algorithm. Dict parameters such as ``sim_options`` require
            special treatment, see :ref:`this note<grid_search_note>`.
        measures(list of string): The performance measures to compute. Allowed
            names are function names as defined in the :mod:`accuracy
            <surprise.accuracy>` module.  Default is ``['rmse', 'mae']``.
        n_jobs(int): The maximum number of algorithm training in parallel.

            - If ``-1``, all CPUs are used.
            - If ``1`` is given, no parallel computing code is used at all,\
                which is useful for debugging.
            - For ``n_jobs`` below ``-1``, ``(n_cpus + n_jobs + 1)`` are\
                used.  For example, with ``n_jobs = -2`` all CPUs but one are\
                used.

            Default is ``1``.
        pre_dispatch(int or string): Controls the number of jobs that get
            dispatched during parallel execution. Reducing this number can be
            useful to avoid an explosion of memory consumption when more jobs
            get dispatched than CPUs can process. This parameter can be:

            - ``None``, in which case all the jobs are immediately created\
                and spawned. Use this for lightweight and fast-running\
                jobs, to avoid delays due to on-demand spawning of the\
                jobs.
            - An int, giving the exact number of total jobs that are\
                spawned.
            - A string, giving an expression as a function of ``n_jobs``,\
                as in ``'2*n_jobs'``.

            Default is ``'2*n_jobs'``.
        seed(int): The value to use as seed for RNG. It will determine how
            splits are defined. If ``None``, the current time since epoch is
            used. Default is ``None``.
        verbose(bool): Level of verbosity. If ``False``, nothing is printed. If
            ``True``, The mean values of each measure are printed along for
            each parameter combination. Default is ``True``.
        joblib_verbose(int): Controls the verbosity of joblib: the higher, the
            more messages.

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
            with `cv_results` that achieved the highest accuracy for that
            measure.
        """

    def __init__(self, algo_class, param_grid, measures=['rmse', 'mae'],
                 n_jobs=1, pre_dispatch='2*n_jobs', seed=None, verbose=1,
                 joblib_verbose=0):
        self.best_params = CaseInsensitiveDefaultDict(list)
        self.best_index = CaseInsensitiveDefaultDict(list)
        self.best_score = CaseInsensitiveDefaultDict(list)
        self.best_estimator = CaseInsensitiveDefaultDict(list)
        self.cv_results = defaultdict(list)
        self.algo_class = algo_class
        self.param_grid = param_grid.copy()
        self.measures = [measure.upper() for measure in measures]
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.seed = seed if seed is not None else int(time.time())
        self.verbose = verbose
        self.joblib_verbose = joblib_verbose

        # As sim_options and bsl_options are dictionaries, they require a
        # special treatment.
        if 'sim_options' in self.param_grid:
            sim_options = self.param_grid['sim_options']
            sim_options_list = [dict(zip(sim_options, v)) for v in
                                product(*sim_options.values())]
            self.param_grid['sim_options'] = sim_options_list

        if 'bsl_options' in self.param_grid:
            bsl_options = self.param_grid['bsl_options']
            bsl_options_list = [dict(zip(bsl_options, v)) for v in
                                product(*bsl_options.values())]
            self.param_grid['bsl_options'] = bsl_options_list

        self.param_combinations = [dict(zip(self.param_grid, v)) for v in
                                   product(*self.param_grid.values())]

        warnings.warn('The GridSearch() class is deprecated. Please use '
                      'model_selection.GridSearchCV instead.', UserWarning)

    def evaluate(self, data):
        """Runs the grid search on dataset.

        Class instance attributes can be accessed after the evaluate is done.

        Args:
            data (:obj:`Dataset <surprise.dataset.Dataset>`): The dataset on
                which to evaluate the algorithm.
        """

        if self.verbose:
            print('Running grid search for the following parameter ' +
                  'combinations:')
            for combination in self.param_combinations:
                print(combination)

        delayed_list = (
            delayed(seed_and_eval)(self.seed,
                                   self.algo_class(**combination),
                                   data,
                                   self.measures)
            for combination in self.param_combinations
        )
        performances_list = Parallel(n_jobs=self.n_jobs,
                                     pre_dispatch=self.pre_dispatch,
                                     verbose=self.joblib_verbose)(delayed_list)

        if self.verbose:
            print('Resulsts:')
        scores = []
        for i, perf in enumerate(performances_list):
            mean_score = {measure: np.mean(perf[measure]) for measure in
                          self.measures}
            scores.append(mean_score)
            if self.verbose:
                print(self.param_combinations[i])
                print(mean_score)
                print('-' * 10)

        # Add all scores and parameters lists to dict
        self.cv_results['params'] = self.param_combinations
        self.cv_results['scores'] = scores

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


def seed_and_eval(seed, *args):
    """Helper function that calls evaluate.evaluate() *after* having seeded
    the RNG. RNG seeding is mandatory since evalute() is called by
    different processes."""

    random.seed(seed)
    return evaluate(*args, verbose=0)
