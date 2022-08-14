
from abc import ABC, abstractmethod
from itertools import product
import numpy as np
from joblib import Parallel
from joblib import delayed

from .split import get_cv
from .validation import fit_and_score
from ..dataset import DatasetUserFolds
from ..utils import get_rng


class BaseSearchCV(ABC):
    """Base class for hyper parameter search with cross-validation."""

    @abstractmethod
    def __init__(self, algo_class, measures=['rmse', 'mae'], cv=None,
                 refit=False, return_train_measures=False, n_jobs=1,
                 pre_dispatch='2*n_jobs', joblib_verbose=0):

        self.algo_class = algo_class
        self.measures = [measure.lower() for measure in measures]
        self.cv = cv

        if isinstance(refit, str):
            if refit.lower() not in self.measures:
                raise ValueError('It looks like the measure you want to use '
                                 'with refit ({}) is not in the measures '
                                 'parameter')

            self.refit = refit.lower()

        elif refit is True:
            self.refit = self.measures[0]

        else:
            self.refit = False

        self.return_train_measures = return_train_measures
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.joblib_verbose = joblib_verbose

    def _parse_options(self, params):
        # As sim_options and bsl_options are dictionaries, they require a
        # special treatment.

        if 'sim_options' in params:
            sim_options = params['sim_options']
            sim_options_list = [dict(zip(sim_options, v)) for v in
                                product(*sim_options.values())]
            params['sim_options'] = sim_options_list

        if 'bsl_options' in params:
            bsl_options = params['bsl_options']
            bsl_options_list = [dict(zip(bsl_options, v)) for v in
                                product(*bsl_options.values())]
            params['bsl_options'] = bsl_options_list

        return params

    def fit(self, data):
        """Runs the ``fit()`` method of the algorithm for all parameter
        combinations, over different splits given by the ``cv`` parameter.

        Args:
            data (:obj:`Dataset <surprise.dataset.Dataset>`): The dataset on
                which to evaluate the algorithm, in parallel.
        """

        if self.refit and isinstance(data, DatasetUserFolds):
            raise ValueError('refit cannot be used when data has been '
                             'loaded with load_from_folds().')

        cv = get_cv(self.cv)

        delayed_list = (
            delayed(fit_and_score)(self.algo_class(**params), trainset,
                                   testset, self.measures,
                                   self.return_train_measures)
            for params, (trainset, testset) in product(self.param_combinations,
                                                       cv.split(data))
        )
        out = Parallel(n_jobs=self.n_jobs,
                       pre_dispatch=self.pre_dispatch,
                       verbose=self.joblib_verbose)(delayed_list)

        (test_measures_dicts,
         train_measures_dicts,
         fit_times,
         test_times) = zip(*out)

        # test_measures_dicts is a list of dict like this:
        # [{'mae': 1, 'rmse': 2}, {'mae': 2, 'rmse': 3} ...]
        # E.g. for 5 splits, the first 5 dicts are for the first param
        # combination, the next 5 dicts are for the second param combination,
        # etc...
        # We convert it into a dict of list:
        # {'mae': [1, 2, ...], 'rmse': [2, 3, ...]}
        # Each list is still of size n_parameters_combinations * n_splits.
        # Then, reshape each list to have 2-D arrays of shape
        # (n_parameters_combinations, n_splits). This way we can easily compute
        # the mean and std dev over all splits or over all param comb.
        test_measures = dict()
        train_measures = dict()
        new_shape = (len(self.param_combinations), cv.get_n_folds())
        for m in self.measures:
            test_measures[m] = np.asarray([d[m] for d in test_measures_dicts])
            test_measures[m] = test_measures[m].reshape(new_shape)
            if self.return_train_measures:
                train_measures[m] = np.asarray([d[m] for d in
                                                train_measures_dicts])
                train_measures[m] = train_measures[m].reshape(new_shape)

        cv_results = dict()
        best_index = dict()
        best_params = dict()
        best_score = dict()
        best_estimator = dict()
        for m in self.measures:
            # cv_results: set measures for each split and each param comb
            for split in range(cv.get_n_folds()):
                cv_results['split{0}_test_{1}'.format(split, m)] = \
                    test_measures[m][:, split]
                if self.return_train_measures:
                    cv_results['split{0}_train_{1}'.format(split, m)] = \
                        train_measures[m][:, split]

            # cv_results: set mean and std over all splits (testset and
            # trainset) for each param comb
            mean_test_measures = test_measures[m].mean(axis=1)
            cv_results['mean_test_{}'.format(m)] = mean_test_measures
            cv_results['std_test_{}'.format(m)] = test_measures[m].std(axis=1)
            if self.return_train_measures:
                mean_train_measures = train_measures[m].mean(axis=1)
                cv_results['mean_train_{}'.format(m)] = mean_train_measures
                cv_results['std_train_{}'.format(m)] = \
                    train_measures[m].std(axis=1)

            # cv_results: set rank of each param comb
            # also set best_index, and best_xxxx attributes
            indices = cv_results['mean_test_{}'.format(m)].argsort()
            cv_results['rank_test_{}'.format(m)] = np.empty_like(indices)
            if m in ('mae', 'rmse', 'mse'):
                cv_results['rank_test_{}'.format(m)][indices] = \
                    np.arange(len(indices)) + 1  # sklearn starts at 1 as well
                best_index[m] = mean_test_measures.argmin()
            elif m in ('fcp',):
                cv_results['rank_test_{}'.format(m)][indices] = \
                    np.arange(len(indices), 0, -1)
                best_index[m] = mean_test_measures.argmax()
            best_params[m] = self.param_combinations[best_index[m]]
            best_score[m] = mean_test_measures[best_index[m]]
            best_estimator[m] = self.algo_class(**best_params[m])

        # Cv results: set fit and train times (mean, std)
        fit_times = np.array(fit_times).reshape(new_shape)
        test_times = np.array(test_times).reshape(new_shape)
        for s, times in zip(('fit', 'test'), (fit_times, test_times)):
            cv_results['mean_{}_time'.format(s)] = times.mean(axis=1)
            cv_results['std_{}_time'.format(s)] = times.std(axis=1)

        # cv_results: set params key and each param_* values
        cv_results['params'] = self.param_combinations
        for param in self.param_combinations[0]:
            cv_results['param_' + param] = [comb[param] for comb in
                                            self.param_combinations]

        if self.refit:
            best_estimator[self.refit].fit(data.build_full_trainset())

        self.best_index = best_index
        self.best_params = best_params
        self.best_score = best_score
        self.best_estimator = best_estimator
        self.cv_results = cv_results

    def test(self, testset, verbose=False):
        """Call ``test()`` on the estimator with the best found parameters
        (according the the ``refit`` parameter). See :meth:`AlgoBase.test()
        <surprise.prediction_algorithms.algo_base.AlgoBase.test>`.

        Only available if ``refit`` is not ``False``.
        """

        if not self.refit:
            raise ValueError('refit is False, cannot use test()')

        return self.best_estimator[self.refit].test(testset, verbose)

    def predict(self, *args):
        """Call ``predict()`` on the estimator with the best found parameters
        (according the the ``refit`` parameter). See :meth:`AlgoBase.predict()
        <surprise.prediction_algorithms.algo_base.AlgoBase.predict>`.

        Only available if ``refit`` is not ``False``.
        """

        if not self.refit:
            raise ValueError('refit is False, cannot use predict()')

        return self.best_estimator[self.refit].predict(*args)


class GridSearchCV(BaseSearchCV):
    """The :class:`GridSearchCV` class computes accuracy metrics for an
    algorithm on various combinations of parameters, over a cross-validation
    procedure. This is useful for finding the best set of parameters for a
    prediction algorithm. It is analogous to `GridSearchCV
    <http://scikit-learn.org/stable/modules/generated/sklearn.
    model_selection.GridSearchCV.html>`_ from scikit-learn.

    See an example in the :ref:`User Guide <tuning_algorithm_parameters>`.

    Args:
        algo_class(:obj:`AlgoBase \
            <surprise.prediction_algorithms.algo_base.AlgoBase>`): The class
            of the algorithm to evaluate.
        param_grid(dict): Dictionary with algorithm parameters as keys and
            list of values as keys. All combinations will be evaluated with
            desired algorithm. Dict parameters such as ``sim_options`` require
            special treatment, see :ref:`this note<grid_search_note>`.
        measures(list of string): The performance measures to compute. Allowed
            names are function names as defined in the :mod:`accuracy
            <surprise.accuracy>` module.  Default is ``['rmse', 'mae']``.
        cv(cross-validation iterator, int or ``None``): Determines how the
            ``data`` parameter will be split (i.e. how trainsets and testsets
            will be defined). If an int is passed, :class:`KFold
            <surprise.model_selection.split.KFold>` is used with the
            appropriate ``n_splits`` parameter. If ``None``, :class:`KFold
            <surprise.model_selection.split.KFold>` is used with
            ``n_splits=5``.
        refit(bool or str): If ``True``, refit the algorithm on the whole
            dataset using the set of parameters that gave the best average
            performance for the first measure of ``measures``. Other measures
            can be used by passing a string (corresponding to the measure
            name). Then, you can use the ``test()`` and ``predict()`` methods.
            ``refit`` can only be used if the ``data`` parameter given to
            ``fit()`` hasn't been loaded with :meth:`load_from_folds()
            <surprise.dataset.Dataset.load_from_folds>`. Default is ``False``.
        return_train_measures(bool): Whether to compute performance measures on
            the trainsets. If ``True``, the ``cv_results`` attribute will
            also contain measures for trainsets. Default is ``False``.
        n_jobs(int): The maximum number of parallel training procedures.

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
        joblib_verbose(int): Controls the verbosity of joblib: the higher, the
            more messages.

    Attributes:
        best_estimator (dict of AlgoBase):
            Using an accuracy measure as key, get the algorithm that gave the
            best accuracy results for the chosen measure, averaged over all
            splits.
        best_score (dict of floats):
            Using an accuracy measure as key, get the best average score
            achieved for that measure.
        best_params (dict of dicts):
            Using an accuracy measure as key, get the parameters combination
            that gave the best accuracy results for the chosen measure (on
            average).
        best_index  (dict of ints):
            Using an accuracy measure as key, get the index that can be used
            with ``cv_results`` that achieved the highest accuracy for that
            measure (on average).
        cv_results (dict of arrays):
            A dict that contains accuracy measures over all splits, as well as
            train and test time for each parameter combination. Can be imported
            into a pandas `DataFrame` (see :ref:`example
            <cv_results_example>`).
    """
    def __init__(self, algo_class, param_grid, measures=['rmse', 'mae'],
                 cv=None, refit=False, return_train_measures=False, n_jobs=1,
                 pre_dispatch='2*n_jobs', joblib_verbose=0):

        super(GridSearchCV, self).__init__(
            algo_class=algo_class, measures=measures, cv=cv, refit=refit,
            return_train_measures=return_train_measures, n_jobs=n_jobs,
            pre_dispatch=pre_dispatch, joblib_verbose=joblib_verbose)

        self.param_grid = self._parse_options(param_grid.copy())
        self.param_combinations = [dict(zip(self.param_grid, v)) for v in
                                   product(*self.param_grid.values())]


class RandomizedSearchCV(BaseSearchCV):
    """The :class:`RandomizedSearchCV` class computes accuracy metrics for an
    algorithm on various combinations of parameters, over a cross-validation
    procedure. As opposed to GridSearchCV, which uses an exhaustive
    combinatorial approach, RandomizedSearchCV samples randomly from the
    parameter space. This is useful for finding the best set of parameters
    for a prediction algorithm, especially using a coarse to fine approach.
    It is analogous to `RandomizedSearchCV <http://scikit-learn.org/stable/
    modules/generated/sklearn.model_selection.RandomizedSearchCV.html>`_ from
    scikit-learn.

    See an example in the :ref:`User Guide <tuning_algorithm_parameters>`.

    Args:
        algo_class(:obj:`AlgoBase \
            <surprise.prediction_algorithms.algo_base.AlgoBase>`): The class
            of the algorithm to evaluate.
        param_distributions(dict): Dictionary with algorithm parameters as
            keys and distributions or lists of parameters to try. Distributions
            must provide a rvs method for sampling (such as those from
            scipy.stats.distributions). If a list is given, it is sampled
            uniformly. Parameters will be sampled n_iter times.
        n_iter(int): Number of times parameter settings are sampled. Default is
            ``10``.
        measures(list of string): The performance measures to compute. Allowed
            names are function names as defined in the :mod:`accuracy
            <surprise.accuracy>` module.  Default is ``['rmse', 'mae']``.
        cv(cross-validation iterator, int or ``None``): Determines how the
            ``data`` parameter will be split (i.e. how trainsets and testsets
            will be defined). If an int is passed, :class:`KFold
            <surprise.model_selection.split.KFold>` is used with the
            appropriate ``n_splits`` parameter. If ``None``, :class:`KFold
            <surprise.model_selection.split.KFold>` is used with
            ``n_splits=5``.
        refit(bool or str): If ``True``, refit the algorithm on the whole
            dataset using the set of parameters that gave the best average
            performance for the first measure of ``measures``. Other measures
            can be used by passing a string (corresponding to the measure
            name). Then, you can use the ``test()`` and ``predict()`` methods.
            ``refit`` can only be used if the ``data`` parameter given to
            ``fit()`` hasn't been loaded with :meth:`load_from_folds()
            <surprise.dataset.Dataset.load_from_folds>`. Default is ``False``.
        return_train_measures(bool): Whether to compute performance measures on
            the trainsets. If ``True``, the ``cv_results`` attribute will
            also contain measures for trainsets. Default is ``False``.
        n_jobs(int): The maximum number of parallel training procedures.

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
        random_state(int, RandomState or None): Pseudo random number
            generator seed used for random uniform sampling from lists of
            possible values instead of scipy.stats distributions. If int,
            ``random_state`` is the seed used by the random number generator.
            If ``RandomState`` instance, ``random_state`` is the random number
            generator. If ``None``, the random number generator is the
            RandomState instance used by ``np.random``.  Default is ``None``.
        joblib_verbose(int): Controls the verbosity of joblib: the higher, the
            more messages.

    Attributes:
        best_estimator (dict of AlgoBase):
            Using an accuracy measure as key, get the algorithm that gave the
            best accuracy results for the chosen measure, averaged over all
            splits.
        best_score (dict of floats):
            Using an accuracy measure as key, get the best average score
            achieved for that measure.
        best_params (dict of dicts):
            Using an accuracy measure as key, get the parameters combination
            that gave the best accuracy results for the chosen measure (on
            average).
        best_index  (dict of ints):
            Using an accuracy measure as key, get the index that can be used
            with ``cv_results`` that achieved the highest accuracy for that
            measure (on average).
        cv_results (dict of arrays):
            A dict that contains accuracy measures over all splits, as well as
            train and test time for each parameter combination. Can be imported
            into a pandas `DataFrame` (see :ref:`example
            <cv_results_example>`).
    """
    def __init__(self, algo_class, param_distributions, n_iter=10,
                 measures=['rmse', 'mae'], cv=None, refit=False,
                 return_train_measures=False, n_jobs=1,
                 pre_dispatch='2*n_jobs', random_state=None, joblib_verbose=0):

        super(RandomizedSearchCV, self).__init__(
            algo_class=algo_class, measures=measures, cv=cv, refit=refit,
            return_train_measures=return_train_measures, n_jobs=n_jobs,
            pre_dispatch=pre_dispatch, joblib_verbose=joblib_verbose)

        self.n_iter = n_iter
        self.random_state = random_state
        self.param_distributions = self._parse_options(
            param_distributions.copy())
        self.param_combinations = self._sample_parameters(
            self.param_distributions, self.n_iter, self.random_state)

    @staticmethod
    def _sample_parameters(param_distributions, n_iter, random_state=None):
        """Samples ``n_iter`` parameter combinations from
        ``param_distributions`` using ``random_state`` as a seed.

        Non-deterministic iterable over random candidate combinations for
        hyper-parameter search. If all parameters are presented as a list,
        sampling without replacement is performed. If at least one parameter
        is given as a distribution, sampling with replacement is used.
        It is highly recommended to use continuous distributions for continuous
        parameters.

        Note that before SciPy 0.16, the ``scipy.stats.distributions`` do not
        accept a custom RNG instance and always use the singleton RNG from
        ``numpy.random``. Hence setting ``random_state`` will not guarantee a
        deterministic iteration whenever ``scipy.stats`` distributions are used
        to define the parameter search space. Deterministic behavior is however
        guaranteed from SciPy 0.16 onwards.

        Args:
            param_distributions(dict): Dictionary where the keys are
                parameters and values are distributions from which a parameter
                is to be sampled. Distributions either have to provide a
                ``rvs`` function to sample from them, or can be given as a list
                 of values, where a uniform distribution is assumed.
            n_iter(int): Number of parameter settings produced.
                Default is ``10``.
            random_state(int, RandomState instance or None):
                Pseudo random number generator seed used for random uniform
                sampling from lists of possible values instead of scipy.stats
                distributions. If ``None``, the random number generator is the
                random state instance used by np.random.  Default is ``None``.

        Returns:
            combos(list): List of parameter dictionaries with sampled values.
        """

        # check if all distributions are given as lists
        # if so, sample without replacement
        all_lists = np.all([not hasattr(v, 'rvs')
                            for v in param_distributions.values()])
        rnd = get_rng(random_state)

        # sort for reproducibility
        items = sorted(param_distributions.items())

        if all_lists:
            # create exhaustive combinations
            param_grid = [dict(zip(param_distributions, v)) for v in
                          product(*param_distributions.values())]
            combos = np.random.choice(param_grid, n_iter, replace=False)

        else:
            combos = []
            for _ in range(n_iter):
                params = dict()
                for k, v in items:
                    if hasattr(v, 'rvs'):
                        params[k] = v.rvs(random_state=rnd)
                    else:
                        params[k] = v[rnd.randint(len(v))]
                combos.append(params)

        return combos
