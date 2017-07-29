from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import defaultdict
import numpy as np

from surprise import Dataset
from surprise import SVD


def precision_recall_at_k(predictions, k=10, threshold=3.5, verbose=True):
    '''Return mean precision and recall at k for all users.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        k(int): The number of recommendations for which precision and recall
            will be computed. Default is 10.
        threshold(float): A value after which ratings are considered positive.
            Default is 3.5
    Returns:
    A tuple with the value (Precision@k, Recall@k)
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append([iid, est, true_r])

    # Create empty array to hold precisions and recalls for all users
    precisions = []
    recalls = []

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings_array in top_n.items():
        user_ratings_array = np.array(user_ratings_array)
        user_ratings_array_sorted = (
                user_ratings_array[user_ratings_array[:, 1].argsort()[::-1]])

        top_k_items = user_ratings_array_sorted[:k]

        user_precision, user_recall = user_precision_recall(
            top_k_items, threshold)
        precisions.append(user_precision)
        recalls.append(user_recall)
    if verbose:
        print(('Precision at %d is %1.4f') % (k, np.mean(precisions)))
        print(('Recall at %d is %1.4f') % (k, np.mean(recalls)))
    return (np.mean(precisions), np.mean(recalls))


def user_precision_recall(top_preds, threshold):
    '''Return mean precision and recall for a specific user.

    Args:
        top_preds(numpy 2d array): An array of [iid, est, true_r]
        threshold(float): A value after which ratings are considered positive.

    Returns:
    A tuple with the value (Precision, Recall)
    '''

    # Count the number of relevant, recommended and
    # (relevant and recommended) items
    is_relevant = np.greater_equal(top_preds[:, 2].astype(float), threshold)
    is_recommended = np.greater_equal(top_preds[:, 1].astype(float), threshold)
    is_relevant_and_recommended = np.logical_and(is_relevant, is_recommended)
    n_rel = np.sum(is_relevant)
    n_rec = np.sum(is_recommended)
    n_rel_and_rec = np.sum(is_relevant_and_recommended)

    # We do not have any false positive (Recommended AND (Not relevant))
    precision = float(n_rel_and_rec) / n_rec if n_rec != 0 else 1

    # We do not have any false negative ((Not recommended) AND (relevant))
    recall = float(n_rel_and_rec) / n_rel if n_rel != 0 else 1
    return(precision, recall)


# First train an SVD algorithm on the movielens dataset.
data = Dataset.load_builtin('ml-100k')

data.split(n_folds=5)
algo = SVD()

for trainset, testset in data.folds():
    algo.train(trainset)
    predictions = algo.test(testset)
    precision_recall_at_k(predictions, k=5, threshold=4)
