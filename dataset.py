from collections import defaultdict

class TrainingData:
    def __init__(self, reader):
        self.rawToInnerIdUsers = {}
        self.rawToInnerIdItems = {}

        currentUIndex = 0
        currentIIndex = 0

        self.rm = {}
        self.ur = defaultdict(list)
        self.ir = defaultdict(list)

        # user raw id, item raw id, rating, time stamp
        for urid, irid, r, timestamp in reader.ratings():
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

            # CAREFUL THERE, IT'S REVERSED
            self.rm[uid, iid] = r
            self.ur[uid].append((iid, r))
            self.ir[iid].append((uid, r))


class MovieLensReader():
    def __init__(self, rawRatings):
        self.rawRatings = rawRatings

    def ratings(self):
        for line in self.rawRatings:
            urid, irid, r, timestamp = line.split()
            yield urid, irid, r, timestamp

f = open('./datasets/ml-100k/u%s.base' % 1, 'r')
mv = MovieLensReader(f)
zob = TrainingData(mv)
print(zob.rm)
