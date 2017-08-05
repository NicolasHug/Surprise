from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import defaultdict

from surprise import Dataset
from surprise import SVD


def precision_recall_at_k(predictions, k=10, threshold=3.5, verbose=True):
    '''Return Precision and recall at K metrics for each user.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        k(int): The number of recommendations on which precision and recall
            will be computed. Default is 10.
        threshold(float): All rating greater than or equal to threshold are
            considered positive. Default is 3.5
    Returns:
    A dict where keys are user (raw) ids and values are tuples
        (precision@k, recall@k).
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    precision_recall_k = defaultdict(tuple)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est, true_r))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)

        # Count number of all relevant items
        true_r_list = [x[2] for x in user_ratings]
        n_rel = len([i for i in true_r_list if i >= threshold])

        # Count number of relevant items in the top-k predictions
        true_r_list_k = [x[2] for x in user_ratings[:k]]
        n_rel_at_k = len([i for i in true_r_list_k if i >= threshold])

        # Precision@K: Proportion of top-k documents that are relevant
        precision_at_k = float(n_rel_at_k)/k

        # Recall@K: Proportion of relevant items that are in the top-k
        recall_at_k = float(n_rel_at_k)/n_rel if n_rel != 0 else 1

        precision_recall_k[uid] = (precision_at_k, recall_at_k)

    return precision_recall_k


data = Dataset.load_builtin('ml-100k')

data.split(n_folds=5)
algo = SVD()

for trainset, testset in data.folds():
    algo.train(trainset)
    predictions = algo.test(testset)
    precision_recall_at_k(predictions, k=5, threshold=4)
