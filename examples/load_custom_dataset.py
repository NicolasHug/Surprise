"""
This module descibes how to load a custom dataset from a single file.

As a custom dataset we will actually use the movielens-100k dataset, but act as
if it were not built-in.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

from surprise import BaselineOnly
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

# path to dataset file
file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
reader = Reader(line_format='user item rating timestamp', sep='\t')

data = Dataset.load_from_file(file_path, reader=reader)

# We can now use this dataset as we please, e.g. calling cross_validate
cross_validate(BaselineOnly(), data, verbose=True)
