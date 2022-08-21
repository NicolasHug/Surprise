"""This module contains built-in datasets that can be automatically
downloaded."""

import errno
import os
import zipfile
from collections import namedtuple

from os.path import join
from urllib.request import urlretrieve


def get_dataset_dir():
    """Return folder where downloaded datasets and other data are stored.
    Default folder is ~/.surprise_data/, but it can also be set by the
    environment variable ``SURPRISE_DATA_FOLDER``.
    """

    folder = os.environ.get(
        "SURPRISE_DATA_FOLDER", os.path.expanduser("~") + "/.surprise_data/"
    )
    try:
        os.makedirs(folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            # reraise exception if folder does not exist and creation failed.
            raise

    return folder


# a builtin dataset has
# - an url (where to download it)
# - a path (where it is located on the filesystem)
# - the parameters of the corresponding reader
BuiltinDataset = namedtuple("BuiltinDataset", ["url", "path", "reader_params"])

BUILTIN_DATASETS = {
    "ml-100k": BuiltinDataset(
        url="https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        path=join(get_dataset_dir(), "ml-100k/ml-100k/u.data"),
        reader_params=dict(
            line_format="user item rating timestamp", rating_scale=(1, 5), sep="\t"
        ),
    ),
    "ml-1m": BuiltinDataset(
        url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        path=join(get_dataset_dir(), "ml-1m/ml-1m/ratings.dat"),
        reader_params=dict(
            line_format="user item rating timestamp", rating_scale=(1, 5), sep="::"
        ),
    ),
    "jester": BuiltinDataset(
        url="https://eigentaste.berkeley.edu/dataset/archive/jester_dataset_2.zip",
        path=join(get_dataset_dir(), "jester/jester_ratings.dat"),
        reader_params=dict(line_format="user item rating", rating_scale=(-10, 10)),
    ),
}


def download_builtin_dataset(name):

    dataset = BUILTIN_DATASETS[name]

    print("Trying to download dataset from " + dataset.url + "...")
    tmp_file_path = join(get_dataset_dir(), "tmp.zip")
    urlretrieve(dataset.url, tmp_file_path)

    with zipfile.ZipFile(tmp_file_path, "r") as tmp_zip:
        tmp_zip.extractall(join(get_dataset_dir(), name))

    os.remove(tmp_file_path)
    print("Done! Dataset", name, "has been saved to", join(get_dataset_dir(), name))
