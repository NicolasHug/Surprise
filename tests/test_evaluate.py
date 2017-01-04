"""
Module for testing the evaluate function.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import tempfile
import shutil

from surprise import NormalPredictor
from surprise import Dataset
from surprise import Reader
from surprise import evaluate


def test_performances():
    """Test the returned dict. Also do dumping."""

    current_dir = os.path.dirname(os.path.realpath(__file__))
    folds_files = [(current_dir + '/custom_train',
                    current_dir + '/custom_test')]

    reader = Reader(line_format='user item rating', sep=' ', skip_lines=3,
                    rating_scale=(1, 5))
    data = Dataset.load_from_folds(folds_files=folds_files, reader=reader)

    algo = NormalPredictor()
    tmp_dir = tempfile.mkdtemp()  # create tmp dir
    performances = evaluate(algo, data, measures=['RmSe', 'Mae'],
                            with_dump=True, dump_dir=tmp_dir, verbose=2)
    shutil.rmtree(tmp_dir)  # remove tmp dir

    assert performances['RMSE'] is performances['rmse']
    assert performances['MaE'] is performances['mae']
