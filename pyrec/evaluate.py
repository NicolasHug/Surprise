import stats
import numpy as np

def evaluate(algo, data):
    rmses = []
    maes = []
    for fold_i, (trainset, testset) in enumerate(data.folds):
        print('-' * 20)
        print('fold ' + str(fold_i))
        algo.train(trainset)
        algo.test(testset)
        rmse, mae = stats.compute_stats(algo.preds)
        rmses.append(rmse)
        maes.append(mae)

    print('-' * 20)
    print('mean RMSE: {0:1.4f}'.format(np.mean(rmses)))
    print('mean MAE : {0:1.4f}'.format(np.mean(maes)))
