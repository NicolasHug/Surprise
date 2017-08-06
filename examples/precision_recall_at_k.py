"""
This module illustrates how to compute Precision at k and Recall at k metrics.
We first train an SVD algorithm on the MovieLens dataset, and then predict all
the ratings for the pairs (user, item) that are not in the training set. We
then compute Precision at k and Recall at k based on user defined k and
threshold values.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import defaultdict

from surprise import Dataset
from surprise import SVD


def precision_recall_at_k(predictions, k=10, threshold=3.5, verbose=True):
    '''Return Precision and recall at k metrics for each user.

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

    # Map the predictions to each user.
    user_est_true = defaultdict(list)
    precision_recall_k = defaultdict(tuple)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    for uid, user_ratings in user_est_true.items():
        # Count number of all relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Sort list by estimated rating
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Count number of relevant items recommended in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold)) for (
                est, true_r) in user_ratings[:k])

        # Count number of recommended items in top k
        n_rec_k = sum(
            (est >= threshold) for (est, _) in user_ratings[:k])

        # Precision@K: Proportion of top-k documents that are relevant
        precision_at_k = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are in the top-k
        recall_at_k = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

        precision_recall_k[uid] = (precision_at_k, recall_at_k)

    return precision_recall_k


data = Dataset.load_builtin('ml-100k')

data.split(n_folds=5)
algo = SVD()

for trainset, testset in data.folds():
    algo.train(trainset)
    predictions = algo.test(testset)
    precision_recall_at_k(predictions, k=5, threshold=4)
