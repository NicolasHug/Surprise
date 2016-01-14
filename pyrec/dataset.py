from collections import defaultdict
from collections import namedtuple
import sys
import os
from urllib.request import urlretrieve
import zipfile
import itertools
import random

# TODO:try to give an explicit error messages if reader fails to parse

Trainset = namedtuple('Trainset',
                     ['rm',
                      'ur',
                      'ir',
                      'n_users',
                      'n_items',
                      'r_min',
                      'r_max',
                      'raw2inner_id_users',
                      'raw2inner_id_items'])


class Dataset:

    def __init__(self, reader=None):

        self.reader = reader

    @classmethod
    def load(cls, name='ml-100k'):
        # hardcode reader and download dataset if needed
        # and then call load_from_file
        pass

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

        with open(file_name) as f:
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

        r_min = 1  #TODO: change that
        r_max = 5

        trainset = Trainset(rm,
                            ur,
                            ir,
                            n_users,
                            n_items,
                            r_min,
                            r_max,
                            raw2inner_id_users,
                            raw2inner_id_items)

        return trainset

    def construct_testset(self, trainset, raw_testset):

        testset = []
        for ruid, riid, r, timestamp in raw_testset:
            try:  #TODO: change that
                uid = trainset.raw2inner_id_users[ruid]
                iid = trainset.raw2inner_id_items[riid]
                testset.append((uid, iid, r))
            except KeyError:
                pass

        return testset


class DatasetUserFolds(Dataset):

    def __init__(self, folds_files=None, reader=None):

        super().__init__(reader)
        self.folds_files = folds_files

    @property
    def raw_folds(self):
        for train_file, test_file in self.folds_files:
            raw_train_ratings = self.read_ratings(train_file)
            raw_test_ratings = self.read_ratings(test_file)
            yield raw_train_ratings, raw_test_ratings


class DatasetAutoFolds(Dataset):

    def __init__(self, ratings_file=None, reader=None):

        super().__init__(reader)
        self.ratings_file = ratings_file
        self.raw_folds = None  # defined by 'split' method

    def split(self, n_folds=5, shuffle=True):

        self.raw_ratings = self.read_ratings(self.ratings_file)
        if shuffle:
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

        self.raw_folds = k_folds(self.raw_ratings, n_folds)


class Reader():

    def __init__(self, line_format, sep, skip_lines=0):
        self.sep = sep
        self.skip_lines = skip_lines

        try:
            splitted_format = line_format.split()
            entities = ('user', 'item', 'rating', 'timestamp')
            self.indexes = [splitted_format.index(entity) for entity in entities]
        except ValueError:
            raise ValueError('Wrong format')

    def read_line(self, line):

        line = line.split(self.sep)
        uid, iid, r, timestamp = (line[i].strip() for i in self.indexes)
        return uid, iid, int(r), timestamp

def download_dataset(name):
    answered = False
    while not answered:
        print('dataset ' + name + ' could not be found. Do you want to '
              'download it? [Y/n]')
        choice = input().lower()
        if choice in ['yes', 'y', '']:
            answered = True
        elif choice in ['no', 'n']:
            answered = True
            print("Ok then, I'm out")
            sys.exit()

    if name == 'ml-100k':
        url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    elif name == 'ml-1m':
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
    elif name == 'BX':
        url = 'http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip'
    elif name == 'jester':
        url = 'http://eigentaste.berkeley.edu/dataset/jester_dataset_2.zip'

    print('downloading...')
    urlretrieve(url, 'tmp.zip')
    print('done')

    # TODO: close zipfile before removing? use a context manager if available
    zf = zipfile.ZipFile('tmp.zip', 'r')
    zf.extractall('datasets/' + name)
    os.remove('tmp.zip')

def get_raw_ratings(name, train_file=None):
    if name == 'ml-100k':
        data_file = train_file or './datasets/ml-100k/ml-100k/u.data'
        reader_klass = MovieLens100kReader
    elif name == 'ml-1m':
        data_file = train_file or './datasets/ml-1m/ml-1m/ratings.dat'
        reader_klass = MovieLens1mReader
    elif name == 'BX':
        reader_klass = BXReader
        data_file = train_file or './datasets/BX/BX-Book-Ratings.csv'
    elif name == 'jester':
        reader_klass = JesterReader
        data_file = train_file or './datasets/jester/jester_ratings.dat'

    if not os.path.isfile(data_file):
        download_dataset(name)

    # open file in latin, else the Book-Ratings dataset raises an utf8 error
    with open(data_file, encoding='latin') as f:
        data = [line for line in f]

    if name == 'BX':
        data = data[1:]  # this really sucks TODO: change that

    return data, reader_klass
