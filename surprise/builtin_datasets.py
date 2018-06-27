'''This module contains built-in datasets that can be automatically
downloaded.'''

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from six.moves.urllib.request import urlretrieve
import zipfile
from collections import namedtuple
import os
from os.path import join


def get_dataset_dir():
    '''Return folder where downloaded datasets and other data are stored.
    Default folder is ~/.surprise_data/, but it can also be set by the
    environment variable ``SURPRISE_DATA_FOLDER``.
    '''

    folder = os.environ.get('SURPRISE_DATA_FOLDER', os.path.expanduser('~') +
                            '/.surprise_data/')
    if not os.path.exists(folder):
        os.makedirs(folder)

    return folder


# a builtin dataset has
# - an url (where to download it)
# - a path (where it is located on the filesystem)
# - a rating scale
# - the parameters of the corresponding reader
BuiltinDataset = namedtuple('BuiltinDataset',
                            ['url', 'path', 'rating_scale', 'reader_params'])

BUILTIN_DATASETS = {
    'ml-100k':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            path=join(get_dataset_dir(), 'ml-100k/ml-100k/u.data'),
            rating_scale=(1, 5),
            reader_params=dict(line_format='user item rating timestamp',
                               sep='\t')
        ),
    'ml-1m':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            path=join(get_dataset_dir(), 'ml-1m/ml-1m/ratings.dat'),
            rating_scale=(1, 5),
            reader_params=dict(line_format='user item rating timestamp',
                               sep='::')
        ),
    'jester':
        BuiltinDataset(
            url='http://eigentaste.berkeley.edu/dataset/jester_dataset_2.zip',
            path=join(get_dataset_dir(), 'jester/jester_ratings.dat'),
            rating_scale=(-10, 10),
            reader_params=dict(line_format='user item rating')
        )
}


def download_builtin_dataset(name):

    dataset = BUILTIN_DATASETS[name]

    print('Trying to download dataset from ' + dataset.url + '...')
    tmp_file_path = join(get_dataset_dir(), 'tmp.zip')
    urlretrieve(dataset.url, tmp_file_path)

    with zipfile.ZipFile(tmp_file_path, 'r') as tmp_zip:
        tmp_zip.extractall(join(get_dataset_dir(), name))

    os.remove(tmp_file_path)
    print('Done! Dataset', name, 'has been saved to',
          join(get_dataset_dir(), name))
