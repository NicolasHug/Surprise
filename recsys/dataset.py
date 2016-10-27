"""
the :mod:`dataset` module defines some tools for managing datasets.

Users may use both *built-in* and user-defined datasets (see the
:ref:`getting_started` page for examples). Right now, four built-in datasets
are available:

* The `movielens-100k <http://grouplens.org/datasets/movielens/>`_ dataset.
* The `movielens-1m <http://grouplens.org/datasets/movielens/>`_ dataset.
* The `Jester <http://eigentaste.berkeley.edu/dataset/>`_ dataset 2.

Built-in datasets can all be loaded (or downloaded if you haven't already)
using the :meth:`Dataset.load_builtin` method. For each built-in dataset,
Recsys also provide predefined :class:`readers <Reader>` which are useful if
you want to use a custom dataset that has the same format as a built-in one.

Summary:

.. autosummary::
    :nosignatures:

    Dataset.load_builtin
    Dataset.load_from_file
    Dataset.load_from_folds
    Dataset.folds
    DatasetAutoFolds.split
    Reader
    Trainset
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import defaultdict
from collections import namedtuple
import sys
import os
import zipfile
import itertools
import random

try:
    from urllib import urlretrieve  # Python 2
except ImportError:
    from urllib.request import urlretrieve  # Python 3


# Again, weird way of creating a named tuple but else the documentation would
# be awful.
class Trainset(namedtuple('Trainset',
                          ['rm', 'ur', 'ir', 'n_users', 'n_items', 'r_min',
                           'r_max', 'raw2inner_id_users',
                           'raw2inner_id_items'])):
    """A named tuple for containing all useful data that constitutes a training
    set.

    Args:
        rm(:obj:`defaultdict` of :obj:`int`): A dictionary containing all known
            ratings.  Keys are tuples (user_id, item_id), values are ratings.
            ``rm`` stands for *ratings matrix*, even though it's not a proper
            matrix object.
        ur(:obj:`defaultdict` of :obj:`list`): A dictionary containing lists of
            tuples of the form ``(item_id, rating)``. Keys are user ids. ``ur``
            stands for *user ratings*.
        ir(:obj:`defaultdict` of :obj:`list`): A dictionary containing lists of
            tuples of the form ``(user_id, rating)``. Keys are item ids. ``ir``
            stands for *item ratings*.
        n_users: Total number of users :math:`|U|`.
        n_items: Total number of items :math:`|I|`.
        r_min: Minimum value of the rating scale.
        r_max: Maximum value of the rating scale.
        raw2inner_id_users(dict): A mapping between raw user id and inner user
            id. See :ref:`this note<raw_inner_note>`.
        raw2inner_id_items(dict): A mapping between raw item id and inner item
            id. See :ref:`this note<raw_inner_note>`.
    """


# directory where builtin datasets are stored. For now it's in the home
# directory under the .recsys_data. May be ask user to define it?
datasets_dir = os.path.expanduser('~') + '/.recsys_data/'
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)


# a builtin dataset has
# - an url (where to download it)
# - a path (where it is located on the filesystem)
# - the parameters of the corresponding reader
BuiltinDataset = namedtuple('BuiltinDataset', ['url', 'path', 'reader_params'])
builtin_datasets = {
    'ml-100k' :
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            path=datasets_dir + 'ml-100k/ml-100k/u.data',
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='\t')
        ),
    'ml-1m' :
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            path=datasets_dir + 'ml-1m/ml-1m/ratings.dat',
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='::')
        ),
    'jester' :
        BuiltinDataset(
            url='http://eigentaste.berkeley.edu/dataset/jester_dataset_2.zip',
            path=datasets_dir + 'jester/jester_ratings.dat',
            reader_params=dict(line_format='user item rating',
                               rating_scale=(-10, 10))
        )
}


class Dataset:
    """Base class for loading datasets.

    Note that you should never instantiate the :class:`Dataset` class directly
    (same goes for its derived classes), but instead use one of the three
    available methods for loading datasets."""

    def __init__(self, reader):

        self.reader = reader
        self.r_min = reader.inf + reader.offset
        self.r_max = reader.sup + reader.offset

    @classmethod
    def load_builtin(cls, name='ml-100k'):
        """Load a built-in dataset.

        If the dataset has not already been loaded, it will be downloaded and
        saved. You will have to split your dataset using the :meth:`split
        <DatasetAutoFolds.split>` method. See an example in the :ref:`User
        Guide <load_builtin_example>`.


        Args:
            name(:obj:`string`): The name of the built-in dataset to load.
                Accepted values are 'ml-100k', 'ml-1m', and 'jester'.
                Default is 'ml-100k'.

        Returns:
            A :obj:`Dataset` object.

        Raises:
            ValueError: If the ``name`` parameter is incorrect.
        """

        try:
            dataset = builtin_datasets[name]
        except KeyError:
            raise ValueError('unknown dataset ' + name +
                             '. Accepted values are ' +
                             ', '.join(builtin_datasets.keys()) + '.')

        # if dataset does not exist, offer to download it
        if not os.path.isfile(dataset.path):
            answered = False
            while not answered:
                print('Dataset ' + name + ' could not be found. Do you want to '
                      'download it? [Y/n] ', end='')
                try:
                    choice = raw_input().lower()
                except:
                    choice = input().lower()

                if choice in ['yes', 'y', '', 'omg this is so nice of you!!']:
                    answered = True
                elif choice in ['no', 'n', 'hell no why would i want that?!']:
                    answered = True
                    print("Ok then, I'm out!")
                    sys.exit()

            print('Trying to download dataset from ' + dataset.url + '...')
            urlretrieve(dataset.url, datasets_dir + 'tmp.zip')

            with zipfile.ZipFile(datasets_dir + 'tmp.zip', 'r') as tmp_zip:
                tmp_zip.extractall(datasets_dir + name)

            os.remove(datasets_dir + 'tmp.zip')
            print('Done! Dataset', name, 'has been saved to',  datasets_dir +
                  name)

        reader = Reader(**dataset.reader_params)

        return cls.load_from_file(file_path=dataset.path, reader=reader)

    @classmethod
    def load_from_file(cls, file_path, reader):
        """Load a dataset from a (custom) file.

        Use this if you want to use a custom dataset and all of the ratings are
        stored in one file. You will have to split your dataset using the
        :meth:`split <DatasetAutoFolds.split>` method. See an example in the
        :ref:`User Guide <load_from_file_example>`.


        Args:
            file_path(:obj:`string`): The path to the file containing ratings.
            reader(:obj:`Reader`): A reader to read the file.
        """

        return DatasetAutoFolds(ratings_file=file_path, reader=reader)

    @classmethod
    def load_from_folds(cls, folds_files, reader):
        """Load a dataset where folds (for cross-validation) are predifined by
        some files.

        The purpose of this method is to cover a common use case where a
        dataset is already split into predefined folds, such as the
        movielens-100k dataset which defines files u1.base, u1.test, u2.base,
        u2.test, etc... It can also be used when you don't want to perform
        cross-validation but still want to specify your training and testing
        data (which comes down to 1-fold cross-validation anyway). See an
        example in the :ref:`User Guide <load_from_folds_example>`.


        Args:
            folds_files(:obj:`iterable` of :obj:`tuples`): The list of the
                folds. A fold is a tuple of the form ``(path_to_train_file,
                path_to_test_file)``.
            reader(:obj:`Reader`): A reader to read the files.

        """

        return DatasetUserFolds(folds_files=folds_files, reader=reader)


    def read_ratings(self, file_name):
        """Return a list of ratings (user, item, rating, timestamp) read from
        file_name"""

        with open(file_name) as f:
            raw_ratings = [self.reader.parse_line(line) for line in
                           itertools.islice(f, self.reader.skip_lines, None)]
        return raw_ratings

    @property
    def folds(self):
        """Generator over the folds of the Dataset.

        See :ref:`User Guide <iterate_over_folds>` for usage.

        Yields:
            tuple: trainset and testset of current fold.
        """

        for raw_trainset, raw_testset in self.raw_folds:
            trainset = self.construct_trainset(raw_trainset)
            testset = self.construct_testset(trainset, raw_testset)
            yield trainset, testset

    def construct_trainset(self, raw_trainset):

        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        rm = defaultdict(int)
        ur = defaultdict(list)
        ir = defaultdict(list)

        # user raw id, item raw id, rating, time stamp
        for urid, irid, r, timestamp in raw_trainset:
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            rm[uid, iid] = r
            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        n_users = len(ur)  # number of users
        n_items = len(ir)  # number of items

        trainset = Trainset(rm,
                            ur,
                            ir,
                            n_users,
                            n_items,
                            self.r_min,
                            self.r_max,
                            raw2inner_id_users,
                            raw2inner_id_items)

        return trainset

    def construct_testset(self, trainset, raw_testset):

        testset = []
        for ruid, riid, r, timestamp in raw_testset:
            # if user or item were not part of the training set, we still add
            # them to testset but they're set to 'unknown'
            try:
                uid = trainset.raw2inner_id_users[ruid]
            except KeyError:
                uid = 'unknown_' + str(ruid)
            try:
                iid = trainset.raw2inner_id_items[riid]
            except KeyError:
                iid = 'unknown_' + str(riid)
            testset.append((uid, iid, r))

        return testset


class DatasetUserFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are predefined."""

    def __init__(self, folds_files=None, reader=None):

        Dataset.__init__(self, reader)
        self.folds_files = folds_files

        # check that all files actually exist.
        for train_test_files in self.folds_files:
            for f in train_test_files:
                if not os.path.isfile(f):
                    raise ValueError('File', f, 'does not exist.')

    @property
    def raw_folds(self):
        for train_file, test_file in self.folds_files:
            raw_train_ratings = self.read_ratings(train_file)
            raw_test_ratings = self.read_ratings(test_file)
            yield raw_train_ratings, raw_test_ratings


class DatasetAutoFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are not predefined. (Or for when there are no folds at
    all)."""

    def __init__(self, ratings_file=None, reader=None):

        Dataset.__init__(self, reader)
        self.ratings_file = ratings_file
        self.n_folds = 5
        self.shuffle = True
        self.raw_ratings = self.read_ratings(self.ratings_file)

    def build_full_trainset(self):
        """Do not split the dataset into folds and just return a trainset as
        is, built from the whole dataset.

        User can then query for predictions, as shown in the :ref:`User Guide
        <train_on_whole_trainset>`.

        Returns:
            The :class:`Trainset`.
        """

        return self.construct_trainset(self.raw_ratings)


    @property
    def raw_folds(self):


        if self.shuffle:
            random.shuffle(self.raw_ratings)
            self.shuffle = False  # set to false for future calls to raw_folds

        def k_folds(seq, n_folds):
            """Inspired from scikit learn KFold method."""

            if n_folds > len(seq) or n_folds < 2:
                raise ValueError('Incorrect value for n_folds.')

            start, stop = 0, 0
            for fold_i in range(n_folds):
                start = stop
                stop += len(seq) // n_folds
                if fold_i < len(seq) % n_folds:
                    stop += 1
                yield seq[:start] + seq[stop:], seq[start:stop]

        return k_folds(self.raw_ratings, self.n_folds)

    def split(self, n_folds=5, shuffle=True):
        """Split the dataset into folds for futur cross-validation.

        If you forget to call :meth:`split`, the dataset will be automatically
        shuffled and split for 5-folds cross-validation.

        You can obtain repeatable splits over your all your experiments by
        seeding the RNG: ::

            import random
            random.seed(my_seed)  # call this before you call split!

        Args:
            n_folds(:obj:`int`): The number of folds.
            shuffle(:obj:`bool`): Whether to shuffle ratings before splitting.
                If ``False``, folds will always be the same each time the
                experiment is run. Default is ``True``.
        """

        self.n_folds = n_folds
        self.shuffle = shuffle


class Reader():
    """The Reader class is used to parse a file containing ratings.

    Such a file is assumed to specify only one rating per line, and each line
    needs to respect the following structure: ::

        user ; item ; rating ; [timestamp]

    where the order of the fields and the seperator (here ';') may be
    arbitrarily defined (see below).  brackets indicate that the timestamp
    field is optional.


    Args:
        name(:obj:`string`, optional): If specified, a Reader for one of the
            built-in datasets is returned and any other parameter is ignored.
            Accepted values are 'ml-100k', 'ml-1m', and 'jester'. Default
            is ``None``.
        line_format(:obj:`string`): The fields names, in the order at which
            they are encountered on a line. Example: ``'item user rating'``.
        sep(char): the separator between fields. Example : ``';'``.
        rating_scale(:obj:`tuple`, optional): The rating scale used for every
            rating.  Default is ``(1, 5)``.
        skip_lines(:obj:`int`, optional): Number of lines to skip at the
            beginning of the file. Default is ``0``.

    """

    def __init__(self, name=None, line_format=None, sep=None, rating_scale=(1, 5),
                 skip_lines=0):

        if name:
            try:
                self.__init__(**builtin_datasets[name].reader_params)
            except KeyError:
                raise ValueError('unknown reader ' + name +
                                 '. Accepted values are ' +
                                 ', '.join(builtin_datasets.keys()) + '.')
        else:
            self.sep = sep
            self.skip_lines = skip_lines
            self.inf, self.sup = rating_scale
            self.offset = -self.inf + 1 if self.inf <= 0 else 0

            splitted_format = line_format.split()

            entities = ['user', 'item', 'rating']
            if 'timestamp' in splitted_format:
                self.with_timestamp = True
                entities.append('timestamp')
            else:
                self.with_timestamp = False

            # check that all fields are correct
            if any(field not in entities for field in splitted_format):
                raise ValueError('line_format parameter is incorrect.')

            self.indexes = [splitted_format.index(entity) for entity in
                            entities]

    def parse_line(self, line):
        '''Parse a line.

        Args:
            line(str): The line to parse

        Returns:
            tuple: User id, item id, rating and timestamp. The timestamp is set
            to ``None`` if it does no exist.
            '''

        line = line.split(self.sep)
        try:
            if self.with_timestamp:
                uid, iid, r, timestamp = (line[i].strip().strip('"')
                                          for i in self.indexes)
            else:
                uid, iid, r = (line[i].strip().strip('"')
                                          for i in self.indexes)
                timestamp = None

        except IndexError:
            raise ValueError(('Impossible to parse line.' +
                             ' Check the line_format  and sep parameters.'))

        return uid, iid, float(r) + self.offset, timestamp
