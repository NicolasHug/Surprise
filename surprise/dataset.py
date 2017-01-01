"""
the :mod:`dataset` module defines some tools for managing datasets.

Users may use both *built-in* and user-defined datasets (see the
:ref:`getting_started` page for examples). Right now, three built-in datasets
are available:

* The `movielens-100k <http://grouplens.org/datasets/movielens/>`_ dataset.
* The `movielens-1m <http://grouplens.org/datasets/movielens/>`_ dataset.
* The `Jester <http://eigentaste.berkeley.edu/dataset/>`_ dataset 2.

Built-in datasets can all be loaded (or downloaded if you haven't already)
using the :meth:`Dataset.load_builtin` method. For each built-in dataset,
Surprise also provide predefined :class:`readers <Reader>` which are useful if
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

import numpy as np
from six.moves import input
from six.moves.urllib.request import urlretrieve
from six.moves import range
from six import iteritems


# directory where builtin datasets are stored. For now it's in the home
# directory under the .surprise_data. May be ask user to define it?
DATASETS_DIR = os.path.expanduser('~') + '/.surprise_data/'

# a builtin dataset has
# - an url (where to download it)
# - a path (where it is located on the filesystem)
# - the parameters of the corresponding reader
BuiltinDataset = namedtuple('BuiltinDataset', ['url', 'path', 'reader_params'])

BUILTIN_DATASETS = {
    'ml-100k':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            path=DATASETS_DIR + 'ml-100k/ml-100k/u.data',
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='\t')
        ),
    'ml-1m':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            path=DATASETS_DIR + 'ml-1m/ml-1m/ratings.dat',
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='::')
        ),
    'jester':
        BuiltinDataset(
            url='http://eigentaste.berkeley.edu/dataset/jester_dataset_2.zip',
            path=DATASETS_DIR + 'jester/jester_ratings.dat',
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
            dataset = BUILTIN_DATASETS[name]
        except KeyError:
            raise ValueError('unknown dataset ' + name +
                             '. Accepted values are ' +
                             ', '.join(BUILTIN_DATASETS.keys()) + '.')

        # if dataset does not exist, offer to download it
        if not os.path.isfile(dataset.path):
            answered = False
            while not answered:
                print('Dataset ' + name + ' could not be found. Do you want '
                      'to download it? [Y/n] ', end='')
                choice = input().lower()

                if choice in ['yes', 'y', '', 'omg this is so nice of you!!']:
                    answered = True
                elif choice in ['no', 'n', 'hell no why would i want that?!']:
                    answered = True
                    print("Ok then, I'm out!")
                    sys.exit()

            if not os.path.exists(DATASETS_DIR):
                os.makedirs(DATASETS_DIR)

            print('Trying to download dataset from ' + dataset.url + '...')
            urlretrieve(dataset.url, DATASETS_DIR + 'tmp.zip')

            with zipfile.ZipFile(DATASETS_DIR + 'tmp.zip', 'r') as tmp_zip:
                tmp_zip.extractall(DATASETS_DIR + name)

            os.remove(DATASETS_DIR + 'tmp.zip')
            print('Done! Dataset', name, 'has been saved to', DATASETS_DIR +
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

        with open(os.path.expanduser(file_name)) as f:
            raw_ratings = [self.reader.parse_line(line) for line in
                           itertools.islice(f, self.reader.skip_lines, None)]
        return raw_ratings

    def folds(self):
        """Generator function to iterate over the folds of the Dataset.

        See :ref:`User Guide <iterate_over_folds>` for usage.

        Yields:
            tuple: :class:`Trainset` and testset of current fold.
        """

        for raw_trainset, raw_testset in self.raw_folds():
            trainset = self.construct_trainset(raw_trainset)
            testset = self.construct_testset(raw_testset)
            yield trainset, testset

    def construct_trainset(self, raw_trainset):

        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        ur = defaultdict(list)
        ir = defaultdict(list)

        # user raw id, item raw id, translated rating, time stamp
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

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        n_users = len(ur)  # number of users
        n_items = len(ir)  # number of items
        n_ratings = len(raw_trainset)

        trainset = Trainset(ur,
                            ir,
                            n_users,
                            n_items,
                            n_ratings,
                            self.reader.rating_scale,
                            self.reader.offset,
                            raw2inner_id_users,
                            raw2inner_id_items)

        return trainset

    def construct_testset(self, raw_testset):

        return [(ruid, riid, r_ui_trans)
                for (ruid, riid, r_ui_trans, _) in raw_testset]


class DatasetUserFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are predefined."""

    def __init__(self, folds_files=None, reader=None):

        Dataset.__init__(self, reader)
        self.folds_files = folds_files

        # check that all files actually exist.
        for train_test_files in self.folds_files:
            for f in train_test_files:
                if not os.path.isfile(os.path.expanduser(f)):
                    raise ValueError('File ' + str(f) + ' does not exist.')

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

    def __init__(self, name=None, line_format=None, sep=None,
                 rating_scale=(1, 5), skip_lines=0):

        if name:
            try:
                self.__init__(**BUILTIN_DATASETS[name].reader_params)
            except KeyError:
                raise ValueError('unknown reader ' + name +
                                 '. Accepted values are ' +
                                 ', '.join(BUILTIN_DATASETS.keys()) + '.')
        else:
            self.sep = sep
            self.skip_lines = skip_lines
            self.rating_scale = rating_scale

            lower_bound, higher_bound = rating_scale
            self.offset = -lower_bound + 1 if lower_bound <= 0 else 0

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

        Ratings are translated so that they are all strictly positive.

        Args:
            line(str): The line to parse

        Returns:
            tuple: User id, item id, rating and timestamp. The timestamp is set
            to ``None`` if it does no exist.
            '''

        line = line.split(self.sep)
        try:
            if self.with_timestamp:
                uid, iid, r, timestamp = (line[i].strip()
                                          for i in self.indexes)
            else:
                uid, iid, r = (line[i].strip()
                               for i in self.indexes)
                timestamp = None

        except IndexError:
            raise ValueError(('Impossible to parse line.' +
                              ' Check the line_format  and sep parameters.'))

        return uid, iid, float(r) + self.offset, timestamp


class Trainset:
    """A trainset contains all useful data that constitutes a training set.

    It is used by the :meth:`train()
    <surprise.prediction_algorithms.algo_base.AlgoBase.train>` method of every
    prediction algorithm. You should not try to built such an object on your
    own but rather use the :meth:`Dataset.folds` method or the
    :meth:`DatasetAutoFolds.build_full_trainset` method.

    Attributes:
        ur(:obj:`defaultdict` of :obj:`list`): The users ratings. This is a
            dictionary containing lists of tuples of the form ``(item_inner_id,
            rating)``. The keys are user inner ids.
        ir(:obj:`defaultdict` of :obj:`list`): The items ratings. This is a
            dictionary containing lists of tuples of the form ``(user_inner_id,
            rating)``. The keys are item inner ids.
        n_users: Total number of users :math:`|U|`.
        n_items: Total number of items :math:`|I|`.
        n_ratings: Total number of ratings :math:`|R_{train}|`.
        rating_scale(tuple): The minimum and maximal rating of the rating
            scale.
        global_mean: The mean of all ratings :math:`\\mu`.
    """

    def __init__(self, ur, ir, n_users, n_items, n_ratings, rating_scale,
                 offset, raw2inner_id_users, raw2inner_id_items):

        self.ur = ur
        self.ir = ir
        self.n_users = n_users
        self.n_items = n_items
        self.n_ratings = n_ratings
        self.rating_scale = rating_scale
        self.offset = offset
        self._raw2inner_id_users = raw2inner_id_users
        self._raw2inner_id_items = raw2inner_id_items
        self._global_mean = None

    def knows_user(self, uid):
        """Indicate if the user is part of the trainset.

        A user is part of the trainset if the user has at least one rating.

        Args:
            uid(int): The (inner) user id. See :ref:`this
                note<raw_inner_note>`.
        Returns:
            ``True`` if user is part of the trainset, else ``False``.
        """

        return uid in self.ur

    def knows_item(self, iid):
        """Indicate if the item is part of the trainset.

        An item is part of the trainset if the item was rated at least once.

        Args:
            iid(int): The (inner) item id. See :ref:`this
                note<raw_inner_note>`.
        Returns:
            ``True`` if item is part of the trainset, else ``False``.
        """

        return iid in self.ir

    def to_inner_uid(self, ruid):
        """Convert a raw **user** id to an inner id.

        See :ref:`this note<raw_inner_note>`.

        Args:
            ruid(str): The user raw id.

        Returns:
            int: The user inner id.

        Raises:
            ValueError: When user is not part of the trainset.
        """

        try:
            return self._raw2inner_id_users[ruid]
        except KeyError:
            raise ValueError(('User ' + str(ruid) +
                              ' is not part of the trainset.'))

    def to_inner_iid(self, riid):
        """Convert a raw **item** id to an inner id.

        See :ref:`this note<raw_inner_note>`.

        Args:
            riid(str): The item raw id.

        Returns:
            int: The item inner id.

        Raises:
            ValueError: When item is not part of the trainset.
        """

        try:
            return self._raw2inner_id_items[riid]
        except KeyError:
            raise ValueError(('Item ' + str(riid) +
                              ' is not part of the trainset.'))

    def all_ratings(self):
        """Generator function to iterate over all ratings.

        Yields:
            A tuple ``(uid, iid, rating)`` where ids are inner ids.
        """

        for u, u_ratings in iteritems(self.ur):
            for i, r in u_ratings:
                yield u, i, r

    def all_users(self):
        """Generator function to iterate over all users.

        Yields:
            Inner id of users.
        """
        return range(self.n_users)

    def all_items(self):
        """Generator function to iterate over all items.

        Yields:
            Inner id of items.
        """
        return range(self.n_items)

    @property
    def global_mean(self):
        """Return the mean of all ratings.

        It's only computed once."""
        if self._global_mean is None:
            self._global_mean = np.mean([r for (_, _, r) in
                                         self.all_ratings()])

        return self._global_mean
