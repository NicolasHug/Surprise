from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import defaultdict
import numpy as np

from surprise import Dataset
from surprise import BaselineOnly


def precision_recall_at_k(predictions, k=10, threshold=3.5, verbose=True):
    '''Return mean precision and recall at k for all users.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        k(int): The number of recommendations on which precision and recall
            will be computed. Default is 10.
        threshold(float): All rating greater than or equal to threshold are
            considered positive. Default is 3.5
    Returns:
    A tuple with the value (Precision@k, Recall@k)
    '''
    # First map the predictions to each user.
    top_n = defaultdict(list)
    precision_recall_k =  defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est, true_r))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        # TODO: Count number of relevant items n_rel
        user_ratings_true_r = [x[2] for x in user_ratings]
        n_rel = len([i for i in user_ratings_true_r if i >= threshold])

        top_n[uid] = user_ratings[:k]
        # TODO: Count number of relevant items n_rel_at_k
        user_ratings_at_k_true_r = [x[2] for x in user_ratings[:k]]
        n_rel_at_k = len([i for i in user_ratings_at_k_true_r if i >= threshold])
        # TODO: Calculate precision as n_rel_rec / k
        precision_at_k = float(n_rel_at_k)/k 
        # TODO: Calculate recall as n_rel_rec / n_rel
        recall_at_k = float(n_rel_at_k)/n_rel if n_rel != 0 else 1
        # TODO: append to the dict an item with values
        precision_recall_k[uid].append((precision_at_k,recall_at_k))
        # TODO: return dict
    return precision_recall_k
    # return top_n
    # # First map the predictions to each user.
    # top_n = defaultdict(list)
    # for uid, iid, true_r, est, _ in predictions:
    #     top_n[uid].append([iid, est, true_r])

    # # Create empty array to hold precisions and recalls for all users
    # precisions = []
    # recalls = []

    # # Then sort the predictions for each user and retrieve the k highest ones.
    # for uid, user_ratings_array in top_n.items():
    #     user_ratings_array = np.array(user_ratings_array)
    #     user_ratings_array_sorted = (
    #             user_ratings_array[user_ratings_array[:, 1].argsort()[::-1]])

    #     top_k_items = user_ratings_array_sorted[:k]

    #     user_precision, user_recall = user_precision_recall(
    #         top_k_items, threshold)
    #     precisions.append(user_precision)
    #     recalls.append(user_recall)
    # if verbose:
    #     print(('Precision at %d is %1.4f') % (k, np.mean(precisions)))
    #     print(('Recall at %d is %1.4f') % (k, np.mean(recalls)))
    # return (np.mean(precisions), np.mean(recalls))


def user_precision_recall(top_preds, threshold):
    '''Return precision and recall for predictions.

    Args:
        top_preds(numpy 2d array): An array of [iid, est, true_r]
        threshold(float): All rating greater than or equal to threshold are
            considered positive. Default is 3.5

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


# # First train an SVD algorithm on the movielens dataset.
# data = Dataset.load_builtin('ml-100k')

# data.split(n_folds=5)
# algo = BaselineOnly()

# for trainset, testset in data.folds():
#     algo.train(trainset)
#     predictions = algo.test(testset)
#     print(precision_recall_at_k(predictions, k=5, threshold=4))

def pred(true_r, est, u0=None):
    """Just a small tool to build a prediction with appropriate format."""
    return (u0, None, true_r, est, None)

predictions = [pred(4, 4.99, u0='u1'), pred(3, 4.98, u0='u1'), pred(4, 4.97, u0='u1'), pred(4, 4.96,
               u0='u1'), pred(4, 4.95, u0='u1'), pred(4, 4.94, u0='u1'), pred(4, 4.93, u0='u1'), pred(3, 4.92,
               u0='u1'), pred(4, 4.91, u0='u1'), pred(3, 4.90, u0='u1'), pred(4, 4.89, u0='u1'), pred(3, 4.88,
               u0='u1'), pred(3, 4.87, u0='u1'), pred(4, 4.86, u0='u1'), pred(3, 4.85, u0='u1'), pred(3, 4.84,
               u0='u1'), pred(3, 4.83, u0='u1'), pred(3, 4.82, u0='u1'), pred(3, 4.81, u0='u1'), pred(4, 4.80,
               u0='u1'), pred(4, 4.79, u0='u1'), pred(4, 4.78, u0='u1'), pred(4, 4.77, u0='u1'), pred(4, 4.76,
               u0='u1'), pred(4, 4.75, u0='u1'), pred(4, 4.74, u0='u1'), pred(4, 4.73, u0='u1'), pred(4, 4.72,
               u0='u1'), pred(4, 4.71, u0='u1'), pred(4, 4.70, u0='u1')]
for i in range(1,11):
    print(precision_recall_at_k(predictions, k=i, threshold=4))