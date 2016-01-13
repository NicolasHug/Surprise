from collections import defaultdict
import sys
import os
from urllib.request import urlretrieve
import zipfile
import itertools

#TODO: convert from raw to inner ids => when? in 'evaluate'?
# try to give an explicit error messages if reader fails to parse
# where do we split raw_ratings for CV? Too soon means we would have tons of
# dupplicate data...

class Dataset:

    def __init__(self, raw_ratings=None, folds=None):
        """Should not be called directly. Use load* constructors instead."""

        self.raw_ratings = raw_ratings
        self.folds = folds

    @classmethod
    def load(cls, name='ml-100k'):
        # hardcode reader and download dataset if needed
        pass

    @classmethod
    def load_from_file(cls, file_name, reader):

        raw_ratings = cls.read_ratings(file_name, reader)
        return cls(raw_ratings)


    @classmethod
    def load_from_files(cls, train_file, test_file, reader):

        folds_files = [(train_file, test_file)]
        return cls.load_from_folds(folds_files, reader)

    @classmethod
    def load_from_folds(cls, folds_files, reader):

        folds = []
        for train_file, test_file in folds_files:
            raw_train_ratings = cls.read_ratings(train_file, reader)
            raw_test_ratings = cls.read_ratings(test_file, reader)
            folds.append((raw_train_ratings, raw_test_ratings))

        return cls(folds=folds)

    @classmethod
    def read_ratings(cls, file_name, reader):
        """Return a list of ratings (user, item, rating, timestamp) read from
        file_name with given reader"""

        with open(file_name) as f:
            ratings_raw = [reader.read_line(line) for line in
                           itertools.islice(f, reader.skip_lines, None)]
        return ratings_raw

    def make_folds(self, n_folds=5):

        if self.folds:
            print("This dataset is already split, I'm not doing anything")
            return

        # else construct the folds...


class Reader():

    def __init__(self, line_format, sep, skip_lines):
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


reader = Reader(line_format='user item rating timestamp', sep='\t',
                skip_lines=19990)

file_name = '/home/nico/dev/pyrec/pyrec/datasets/ml-100k/ml-100k/u1.test'
Dataset.load_from_files(train_file, test_file, reader)


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
