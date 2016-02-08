import stats
import numpy as np

def evaluate(algo, data):
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

    print('-' * 20)
    print('mean RMSE: {0:1.4f}'.format(np.mean(rmses)))
    print('mean MAE : {0:1.4f}'.format(np.mean(maes)))
    print('mean FCP : {0:1.4f}'.format(np.mean(fcps)))
