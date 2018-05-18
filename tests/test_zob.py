from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os

from surprise import Reader
from surprise import Dataset

def test_i_dont_know():

    reader = Reader(line_format='user item rating', sep=' ',
                    rating_scale=(-10, 10))
    data_file = (os.path.dirname(os.path.realpath(__file__)) +
                 '/custom_dataset_like_jester')
    data = Dataset.load_from_file(data_file, reader)
    print(data.raw_ratings)
