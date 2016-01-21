import stats
import numpy as np

def evaluate(algo, data):
    rmses = []
    maes = []
    fcps = []
    for fold_i, (trainset, testset) in enumerate(data.folds):
        print('-' * 20)
        print('fold ' + str(fold_i))
        algo.train(trainset)
        algo.test(testset)
        rmse, mae = stats.get_rmse_mae(algo.preds)
        rmses.append(rmse)
        maes.append(mae)
        fcps.append(stats.get_FCP(algo.preds))

    print('-' * 20)
    print('mean RMSE: {0:1.4f}'.format(np.mean(rmses)))
    print('mean MAE : {0:1.4f}'.format(np.mean(maes)))
    print('mean FCP : {0:1.4f}'.format(np.mean(fcps)))
