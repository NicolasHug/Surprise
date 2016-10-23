"""
The :mod:`evaluate` module defines the :func:`evaluate` function.
"""

from statistics import mean
import pickle
import time
import os

from . import accuracy

#TODO: the accuracy measures should be chosen by user
def evaluate(algo, data, with_dump=False, verbose=1):
    """Evaluate the performance of the algorithm on given data.

    Depending on the nature of the ``data`` parameter, it may or may not
    perform cross validation.

    Args:
        algo(:obj:`AlgoBase <pyrec.prediction_algorithms.bases.AlgoBase>`): The
            algorithm to evaluate.
        data(:obj:`Dataset <pyrec.dataset.Dataset>`): The dataset on which to
            evaluate the algorithm.
        with_dump(bool): If True, the algorithm parameters and every prediction
            prediction will be dumped (using `Pickle
            <https://docs.python.org/3/library/pickle.html>`_) for potential
            further analysis. Default is ``False``.
        verbose(int): Level of verbosity. If 0, nothing is printed. If 1
            (default), accuracy measures for each folds are printed, with a
            final summary. If 2, every prediction is printed.

    Return:
        Three lists containing RMSE, MAE and FCP evaluations on each fold.
    """

    dump = {}
    rmses = []
    maes = []
    fcps = []

    for fold_i, (trainset, testset) in enumerate(data.folds):

        if verbose:
            print('-' * 20)
            print('fold ' + str(fold_i))

        # train and test algorithm. Keep all rating predictions in a list
        algo.train(trainset)
        predictions = algo.test(testset, verbose=(verbose==2))

        # compute needed performance statistics
        rmses.append(accuracy.rmse(predictions, verbose=verbose))
        maes.append(accuracy.mae(predictions, verbose=verbose))
        fcps.append(accuracy.fcp(predictions, verbose=verbose))

        if with_dump:
            fold_dump = dict(trainset=trainset, predictions=predictions)
            dump['fold_' + str(fold_i)] = fold_dump

    if verbose:
        print('-' * 20)
        print('mean RMSE: {0:1.4f}'.format(mean(rmses)))
        print('mean MAE : {0:1.4f}'.format(mean(maes)))
        print('mean FCP : {0:1.4f}'.format(mean(fcps)))

    if with_dump:
        dump['user_based'] = algo.user_based
        dump['algo'] = algo.__class__.__name__
        dump_evaluation(dump)

    return rmses, maes, fcps


def dump_evaluation(dump):

    dump_dir = os.path.expanduser('~') + '/.pyrec_data/dumps/'

    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    date = time.strftime('%y%m%d-%Hh%Mm%S', time.localtime())
    name = (dump_dir + date + '-' + dump['algo'])

    pickle.dump(dump, open(name, 'wb'))
    print('File has been dumped to', dump_dir)
