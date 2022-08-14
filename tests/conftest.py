"""
This module contains the pytest fixtures.
"""



import os

import pytest

from surprise import Reader
from surprise import Dataset
from surprise.model_selection import PredefinedKFold


@pytest.fixture
def toy_data_reader():
    return Reader(line_format='user item rating', sep=' ', skip_lines=3,
                  rating_scale=(1, 5))


@pytest.fixture
def toy_data(toy_data_reader):

    toy_data_path = (os.path.dirname(os.path.realpath(__file__)) +
                     '/custom_dataset')
    data = Dataset.load_from_file(file_path=toy_data_path,
                                  reader=toy_data_reader)

    return data


@pytest.fixture
def u1_ml100k():
    """Return a Dataset object that contains 10% of the u1 fold from movielens
    100k. Trainset has 8000 ratings and testset has 2000.
    """
    train_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
    test_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
    data = Dataset.load_from_folds([(train_file, test_file)],
                                   Reader('ml-100k'))

    return data


@pytest.fixture
def small_ml():
    """Return a Dataset object with 2000 movielens-100k ratings.
    """
    data_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_test')
    data = Dataset.load_from_file(data_file, Reader('ml-100k'))

    return data


@pytest.fixture
def pkf():
    return PredefinedKFold()
