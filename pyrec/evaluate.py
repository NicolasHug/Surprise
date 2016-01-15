def evaluate(algo, data):
    for trainset, testset in data.folds:
        algo.train(trainset)
        algo.test(testset)
        stats.compute_stats(algo.preds)
