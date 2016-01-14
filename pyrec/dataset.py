from collections import defaultdict
import sys
import os
from urllib.request import urlretrieve
import zipfile
import itertools
import random

# TODO:try to give an explicit error messages if reader fails to parse


class Dataset:

    def __init__(self, reader=None):

        self.reader = reader
        self.raw_ratings = None  # generator defined in subclasses

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
                           itertools.islice(f, reader.skip_lines, None)]
        return raw_ratings

    @property
    def folds(self):

        if not self.raw_folds:
            raise ValueError("raw_folds is unset. Are you sure your dataset " +
                             "is split?")

        for raw_trainset, raw_testset in self.raw_folds:
            print(len(raw_trainset), len(raw_testset))


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
        self.raw_folds = None  # set in 'split' method

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
        return uid, iid, r, timestamp


reader = Reader(line_format='user item rating timestamp', sep='\t')

train_name = '/home/nico/dev/pyrec/pyrec/datasets/ml-100k/ml-100k/u1.base'
test_name = '/home/nico/dev/pyrec/pyrec/datasets/ml-100k/ml-100k/u1.test'
data = Dataset.load_from_files(train_name, test_name, reader=reader)

"""
file_name = '/home/nico/dev/pyrec/pyrec/datasets/ml-100k/ml-100k/u1.test'
data = Dataset.load_from_file(file_name, reader)
data.split(n_folds=5)
"""

data.folds

class TrainingData:

    def __init__(self, reader):

        self.r_max = reader.r_max
        self.r_min = reader.r_min

        self.raw2inner_id_users = {}
        self.raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        self.rm = defaultdict(int)
        self.ur = defaultdict(list)
        self.ir = defaultdict(list)

        # user raw id, item raw id, rating, time stamp
        for urid, irid, r, timestamp in reader.ratings:
            try:
                uid = self.raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                self.raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = self.raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                self.raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            self.rm[uid, iid] = r
            self.ur[uid].append((iid, r))
            self.ir[iid].append((uid, r))

        self.n_users = len(self.ur)  # number of users
        self.n_items = len(self.ir)  # number of items



class MovieLensReader(Reader):

    def __init__(self, raw_ratings):
        super().__init__(raw_ratings)
        self.r_min, self.r_max = (1, 5)


class MovieLens100kReader(MovieLensReader):

    def __init__(self, raw_ratings):
        super().__init__(raw_ratings)

    @property
    def ratings(self):
        for line in self.raw_ratings:
            urid, irid, r, timestamp = line.split()
            yield int(urid), int(irid), int(r), timestamp


class MovieLens1mReader(MovieLensReader):

    def __init__(self, raw_ratings):
        super().__init__(raw_ratings)

    @property
    def ratings(self):
        for line in self.raw_ratings:
            urid, irid, r, timestamp = line.split('::')
            yield int(urid), int(irid), int(r), timestamp


class BXReader(Reader):

    def __init__(self, raw_ratings):
        super().__init__(raw_ratings)
        # implicit info (null rating) is discarded
        self.r_min, self.r_max = (1, 10)

    @property
    def ratings(self):
        for line in self.raw_ratings:
            urid, irid, r = line[:-1].split(';')
            yield urid[1:-1], irid[1:-1], int(r[1:-1]), 0


class JesterReader(Reader):

    def __init__(self, raw_ratings):
        super().__init__(raw_ratings)
        # raw ratings are in [-10, 10]. We need to offset the of 11 so that
        # zero correponds to 'unrated'
        self.r_min, self.r_max = (1, 21)

    @property
    def ratings(self):
        for line in self.raw_ratings:
            urid, irid, r = line.split()
            yield int(urid), int(irid), float(r) + 11, 0


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
