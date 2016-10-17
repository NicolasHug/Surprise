from statistics import mean
import pickle
import time
import os

from . import stats

def evaluate(algo, data, with_dump=False):

    dump = {}
    rmses = []
    maes = []
    fcps = []

    for fold_i, (trainset, testset) in enumerate(data.folds):
        print('-' * 20)
        print('fold ' + str(fold_i))

        # train and test algorithm. Keep all rating predictions in a list
        algo.train(trainset)
        predictions = algo.test(testset)

        # compute needed performance statistics
        rmses.append(stats.rmse(predictions))
        maes.append(stats.mae(predictions))
        fcps.append(stats.fcp(predictions))

        if with_dump:
            fold_dump = dict(trainset=trainset, predictions=predictions)
            dump['fold_' + str(fold_i)] = fold_dump

    print('-' * 20)
    print('mean RMSE: {0:1.4f}'.format(mean(rmses)))
    print('mean MAE : {0:1.4f}'.format(mean(maes)))
    print('mean FCP : {0:1.4f}'.format(mean(fcps)))

    if with_dump:
        dump['user_based'] = algo.user_based
        dump['algo'] = algo.__class__.__name__
        dump_evaluation(dump)


def dump_evaluation(dump):

    dump_dir = os.path.expanduser('~') + '/.pyrec_data/dumps/'

    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    date = time.strftime('%y%m%d-%Hh%Mm%S', time.localtime())
    name = (dump_dir + date + '-' + dump['algo'])

    pickle.dump(dump, open(name, 'wb'))
    print('File has been dumped to', dump_dir)
