"""
Module for testing the rating_scale parameter.

Up until 1.0.6 it was part of the Reader object, but it does not make sense, so
we now make it part of Dataset. If it's not given at one of the Datasets
constructors, we try to switch back to the Reader object with a deprecation
warning.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

import pytest
import pandas as pd

from surprise import Reader
from surprise import Dataset


def test_deprecated_way():
    """Test all Dataset constructors without passing rating_scale as a
    parameter. Make sure we revert back to the Reader object, with a warning
    message.

    Also, make sure ValueError is raised if reader has no rating_scale in this
    context.

    Not using dataset fixtures here for more control.
    """

    # test load_from_file
    toy_data_path = (os.path.dirname(os.path.realpath(__file__)) +
                     '/custom_dataset')
    with pytest.warns(UserWarning):
        reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                        rating_scale=(1, 5))
        data = Dataset.load_from_file(file_path=toy_data_path,
                                      reader=reader)

    with pytest.raises(ValueError):
        reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                        rating_scale=None)
        data = Dataset.load_from_file(file_path=toy_data_path,
                                      reader=reader)

    # test load_from_folds
    train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
    test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
    with pytest.warns(UserWarning):
        reader = Reader(line_format='user item rating timestamp', sep='\t',
                        rating_scale=(1, 5))
        data = Dataset.load_from_folds([(train_file, test_file)], reader=reader)
    with pytest.raises(ValueError):
        reader = Reader(line_format='user item rating timestamp', sep='\t',
                        rating_scale=None)
        data = Dataset.load_from_folds([(train_file, test_file)],
                                       reader=reader)
    # test load_from_df
    ratings_dict = {'itemID': [1, 1, 1, 2, 2],
                    'userID': [9, 32, 2, 45, '10000'],
                    'rating': [3, 2, 4, 3, 1]}
    df = pd.DataFrame(ratings_dict)

    with pytest.warns(UserWarning):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']],
                                    reader=reader)
    with pytest.raises(ValueError):
        reader = Reader(rating_scale=None)
        data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']],  # noqa
                                    reader=reader)
