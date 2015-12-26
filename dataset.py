from collections import defaultdict

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
