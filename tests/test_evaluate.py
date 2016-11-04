"""
Module for testing the evaluate function.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

from recsys import NormalPredictor
from recsys.dataset import Dataset
from recsys.dataset import Reader
from recsys.evaluate import evaluate


def test_performances():
    """Test the returned dict."""

    current_dir = os.path.dirname(os.path.realpath(__file__))
    folds_files = [(current_dir + '/custom_train',
                    current_dir + '/custom_test')]

    reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                    rating_scale=(1, 5))
    data = Dataset.load_from_folds(folds_files=folds_files, reader=reader)

    algo = NormalPredictor()
    performances = evaluate(algo, data, measures=['RmSe', 'Mae'])

    assert performances['RMSE'] is performances['rmse']
    assert performances['MaE'] is performances['mae']
