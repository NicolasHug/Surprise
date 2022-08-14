"""
This module contains tests to make sure that rating scales like that of Jester
(i.e. [-10, 10]) are correctly handled.  It was introduced since we removed the
offset field in Reader, Trainset, etc.

Some context:

The original reason behind the offset field was to remap ratings in (e.g.) [-5,
5] to ratings in [1, 11], to avoid having 0 ratings. This could cause problems
if we were storing ratings in a sparse matrix: there would be no distinction
between 0 ratings and missing ones.

We currently store ratings as two defaultdict of lists (trainset.ur and
trainset.ir), so 0 ratings should be handled correctly, without needing to
remap them. Maybe it will change in the future, we would have to find a
workaround.
"""




import pandas as pd

from surprise import Reader
from surprise import Dataset


def test_zero_rating_canary():

    reader = Reader(rating_scale=(-10, 10))

    ratings_dict = {'itemID': [0, 0, 0, 0, 1, 1],
                    'userID': [0, 1, 2, 3, 3, 4],
                    'rating': [-10, 10, 0, -5, 0, 5]}
    df = pd.DataFrame(ratings_dict)
    data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
    trainset = data.build_full_trainset()

    # test ur and ir fields. Kind of OK, but the purpose of the test is
    # precisely to test what would happen if we removed them...
    assert trainset.ir[0] == [(0, -10), (1, 10), (2, 0), (3, -5)]
    assert trainset.ir[1] == [(3, 0), (4, 5)]

    assert trainset.ur[0] == [(0, -10)]
    assert trainset.ur[1] == [(0, 10)]
    assert trainset.ur[2] == [(0, 0)]
    assert trainset.ur[3] == [(0, -5), (1, 0)]
    assert trainset.ur[4] == [(1, 5)]
    print(trainset.ur)

    # ... so also test all_ratings which should be more reliable.
    all_ratings = list(trainset.all_ratings())
    assert (0, 0, -10) in all_ratings
    assert (1, 0, 10) in all_ratings
    assert (2, 0, 0) in all_ratings
    assert (3, 0, -5) in all_ratings
    assert (3, 1, 0) in all_ratings
    assert (4, 1, 5) in all_ratings
