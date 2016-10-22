"""
Module for testing the Dataset class
"""

import os
from pyrec import Dataset
from pyrec import Reader


reader = Reader(line_format='user item rating', sep=' ', skip_lines=3)
file_path = os.path.dirname(os.path.realpath(__file__)) + '/custom_dataset'

def test_split():
    data = Dataset.load_from_file(file_path=file_path, reader=reader)
    data.split(5)
    assert len(list(data.folds)) == 5

# TODO: ensure that folds are the same when split is called only once
