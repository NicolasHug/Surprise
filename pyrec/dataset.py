"""
the :mod:`dataset` module defines some tools for managing datasets.
"""

from collections import defaultdict
from collections import namedtuple
import sys
import os
from urllib.request import urlretrieve
import zipfile
import itertools
import random

# TODO: try to give an explicit error messages if reader fails to parse
# TODO: change name 'rm' ? it used to mean ratings matrix but now it's a
# dict...
# TODO: Raw2innerId stuff ? Is it usefull to keep it in the Trainset ?? 
# TODO: Plus, is it useful at all to make the mapping as we are now using
# dictionnaries for storing ratings?

# Again, weird way of creating a named tuple but else the documentation would
# be awful.
class Trainset(namedtuple('Trainset',
                          ['rm', 'ur', 'ir', 'n_users', 'n_items', 'r_min',
                           'r_max', 'raw2inner_id_users',
                           'raw2inner_id_items'])):
    """A named tuple for containing all useful data that constitutes a training
    set.

    Args:
        rm(defaultdict of int): A dictionary containing containing all known ratings.
            Keys are tuples (user_id, item_id), values are ratings.
        ur(defaultdict of list): A dictionary containing lists of tuples of the
            form (item_id, rating). Keys are user ids.
        ir(defaultdict of list): A dictionary containing lists of tuples of the
            form (user_id, rating). Keys are item ids.
        n_users: Total number of users :math:`|U|`.
        n_items: Total number of items :math:`|I|`.
        r_min: Minimum value of the rating scale.
        r_max: Maximum value of the rating scale.
        raw2inner_id_users(dict): A mapping between raw user id and inner user
            id. A raw id is an id as written on a dataset file, e.g. for the BX
            dataset it might be '034545104X'. An inner id is an integer from 0
            to n_users, which is a lot more convenient to manage.
        raw2inner_id_items(dict): A mapping between raw item id and inner item
            id. See previous note on `raw2inner_id_users` parameter.

    """


# directory where builtin datasets are stored. For now it's in the home
# directory under the .pyrec_data. May be ask user to define it?
datasets_dir = os.path.expanduser('~') + '/.pyrec_data/'

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
                               interval=(1, 5),
                               sep='\t')
        ),
    'ml-1m' :
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            path=datasets_dir + 'ml-1m/ml-1m/ratings.dat',
            reader_params=dict(line_format='user item rating timestamp',
                               interval=(1, 5),
                               sep='::')
        ),
    'BX' :  # Note that implicit ratings are discarded
        BuiltinDataset(
            url='http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip',
            path=datasets_dir + 'BX/BX-Book-Ratings.csv',
            reader_params=dict(line_format='user item rating',
                               sep=';',
                               interval=(1, 10),
                               skip_lines=1)
        ),
    'jester' :
        BuiltinDataset(
            url='http://eigentaste.berkeley.edu/dataset/jester_dataset_2.zip',
            path=datasets_dir + 'jester/jester_ratings.dat',
            reader_params=dict(line_format='user item rating',
                               interval=(-10, 10))
        )
}


class Dataset:
    """TODO base class for subclasses"""

    def __init__(self, reader):

        self.reader = reader
        self.r_min = reader.inf + reader.offset
        self.r_max = reader.sup + reader.offset

    @classmethod
    def load(cls, name='ml-100k'):

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
                choice = input().lower()
                if choice in ['yes', 'y', '', 'omg this is so nice of you!!']:
                    answered = True
                elif choice in ['no', 'n', 'hell no why would i want that?!']:
                    answered = True
                    print("Ok then, I'm out!")
                    sys.exit()

            print('Trying to download dataset from ' + dataset.url + '...')
            urlretrieve(dataset.url, 'tmp.zip')

            with zipfile.ZipFile('tmp.zip', 'r') as tmp_zip:
                tmp_zip.extractall(datasets_dir + name)

            os.remove('tmp.zip')
            print('Done! Dataset', name, 'has been saved to',  datasets_dir +
                  name)

        reader = Reader(**dataset.reader_params)

        return cls.load_from_file(file_name=dataset.path, reader=reader)

    @classmethod
    def load_from_file(cls, file_name, reader):

        return DatasetAutoFolds(ratings_file=file_name, reader=reader)

    @classmethod
    def load_from_files(cls, train_file, test_file, reader):

        folds_files = [(train_file, test_file)]
        return cls.load_from_folds(folds_files, reader)

    @classmethod
    def load_from_folds(cls, folds_files, reader):

        return DatasetUserFolds(folds_files=folds_files, reader=reader)


    def read_ratings(self, file_name):
        """Return a list of ratings (user, item, rating, timestamp) read from
        file_name"""

        with open(file_name, errors='replace') as f:
            raw_ratings = [self.reader.read_line(line) for line in
                           itertools.islice(f, self.reader.skip_lines, None)]
        return raw_ratings

    @property
    def folds(self):

        if not self.raw_folds:
            raise ValueError("raw_folds is unset. Are you sure your dataset " +
                             "is split?")

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
                uid = 'unknown'
            try:
                iid = trainset.raw2inner_id_items[riid]
            except KeyError:
                iid = 'unknown'
            testset.append((uid, iid, r))

        return testset


class DatasetUserFolds(Dataset):
    """TODO"""

    def __init__(self, folds_files=None, reader=None):

        Dataset.__init__(self, reader)
        self.folds_files = folds_files

    # TODO: as raw_folds and folds are generator, files are only opened and
    # read when needed. It might be good idea to check at least if they all
    # exist at the beggining, so that the program does not crash on the 10th
    # fold...

    @property
    def raw_folds(self):
        for train_file, test_file in self.folds_files:
            raw_train_ratings = self.read_ratings(train_file)
            raw_test_ratings = self.read_ratings(test_file)
            yield raw_train_ratings, raw_test_ratings


class DatasetAutoFolds(Dataset):
    """TODO"""

    def __init__(self, ratings_file=None, reader=None):

        Dataset.__init__(self, reader)
        self.ratings_file = ratings_file
        self.n_folds = 5
        self.shuffle = True

    @property
    def raw_folds(self):

        self.raw_ratings = self.read_ratings(self.ratings_file)

        if self.shuffle:
            random.shuffle(self.raw_ratings)

        def k_folds(seq, n_folds):
            """Inspired from scikit learn KFold method."""
            start, stop = 0, 0
            for fold_i in range(n_folds):
                start = stop
                stop += len(seq) // n_folds
                if fold_i < len(seq) % n_folds:
                    stop += 1
                yield seq[:start] + seq[stop:], seq[start:stop]

        return k_folds(self.raw_ratings, self.n_folds)

    def split(self, n_folds, shuffle=True):
        """Split da dataset yo"""

        self.n_folds = n_folds
        self.shuffle = shuffle


class Reader():

    def __init__(self, name=None, line_format=None, sep=None, interval=(1, 5),
                 skip_lines=0):

        # TODO: I'm not sure this is a nice way to handle a builtin
        # constructor... needs to be checked
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
            self.inf, self.sup = interval
            self.offset = -self.inf + 1 if self.inf <= 0 else 0

            try:
                splitted_format = line_format.split()

                entities = ['user', 'item', 'rating']
                if 'timestamp' in splitted_format:
                    self.with_timestamp = True
                    entities.append('timestamp')
                else:
                    self.with_timestamp = False

                self.indexes = [splitted_format.index(entity) for entity in
                                entities]
            except ValueError:
                raise ValueError('Wrong format')

    def read_line(self, line):

        line = line.split(self.sep)
        uid, iid, *remaining = (line[i].strip().strip('"') for i in self.indexes)
        if self.with_timestamp:
            r, timestamp = remaining
        else:
            r, timestamp = *remaining, None

        return uid, iid, float(r) + self.offset, timestamp
