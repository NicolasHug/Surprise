from collections import defaultdict
import sys
import os
from urllib.request import urlretrieve
import zipfile

class TrainingData:
    def __init__(self, reader):

        self.rMax = reader.rMax
        self.rMin = reader.rMin

        self.rawToInnerIdUsers = {}
        self.rawToInnerIdItems = {}

        currentUIndex = 0
        currentIIndex = 0

        self.rm = defaultdict(int)
        self.ur = defaultdict(list)
        self.ir = defaultdict(list)

        # user raw id, item raw id, rating, time stamp
        for urid, irid, r, timestamp in reader.ratings:
            try:
                uid = self.rawToInnerIdUsers[urid]
            except KeyError:
                uid = currentUIndex
                self.rawToInnerIdUsers[urid] = currentUIndex
                currentUIndex += 1
            try:
                iid = self.rawToInnerIdItems[irid]
            except KeyError:
                iid = currentIIndex
                self.rawToInnerIdItems[irid] = currentIIndex
                currentIIndex += 1

            self.rm[uid, iid] = r
            self.ur[uid].append((iid, r))
            self.ir[iid].append((uid, r))

        self.nUsers = len(self.ur) # number of users
        self.nItems = len(self.ir) # number of items


class Reader():
    def __init__(self, rawRatings):
        self.rawRatings = rawRatings

class MovieLensReader(Reader):
    def __init__(self, rawRatings):
        super().__init__(rawRatings)
        self.rMin, self.rMax = (1, 5)

class MovieLens100kReader(MovieLensReader):
    def __init__(self, rawRatings):
        super().__init__(rawRatings)

    @property
    def ratings(self):
        for line in self.rawRatings:
            urid, irid, r, timestamp = line.split()
            yield int(urid), int(irid), int(r), timestamp

class MovieLens1mReader(MovieLensReader):
    def __init__(self, rawRatings):
        super().__init__(rawRatings)

    @property
    def ratings(self):
        for line in self.rawRatings:
            urid, irid, r, timestamp = line.split('::')
            yield int(urid), int(irid), int(r), timestamp

class BXReader(Reader):
    def __init__(self, rawRatings):
        super().__init__(rawRatings)
        # implicit info (null rating) is discarded
        self.rMin, self.rMax = (1, 10)

    @property
    def ratings(self):
        for line in self.rawRatings:
            urid, irid, r = line.split(';')
            yield int(urid), int(irid), int(r), 0

class JesterReader(Reader):
    def __init__(self, rawRatings):
        super().__init__(rawRatings)
        # raw ratings are in [-10, 10]. We need to offset the of 11 so that
        # zero correponds to 'unrated'
        self.rMin, self.rMax = (1, 21)

    @property
    def ratings(self):
        # TODO: for now only consider the beggining of the file because, else
        # Python throws MemoryError.
        for line in self.rawRatings[:10000]:
            urid, irid, r = line.split()
            yield int(urid), int(irid), float(r) + 11, 0

def downloadDataset(name):
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

    zf = zipfile.ZipFile('tmp.zip', 'r')
    zf.extractall('datasets/' + name)
    os.remove('tmp.zip')

def getRawRatings(name, trainFile=None):
    if name == 'ml-100k':
        dataFile = trainFile or './datasets/ml-100k/ml-100k/u.data'
        ReaderClass = MovieLens100kReader
    elif name == 'ml-1m':
        dataFile = trainFile or './datasets/ml-1m/ml-1m/ratings.dat'
        ReaderClass = MovieLens1mReader
    elif name == 'BX':
        ReaderClass = BXReader
        dataFile = trainFile or './datasets/BX/BX-Book-Ratings.csv'
    elif name =='jester':
        ReaderClass = JesterReader
        dataFile = trainFile or './datasets/jester/jester_ratings.dat'

    try:
        f = open(dataFile, 'r')
    except FileNotFoundError:
        downloadDataset(name)
        f = open(dataFile, 'r')

    data = [line for line in f]

    return data, ReaderClass
