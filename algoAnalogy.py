import random as rd

from algo import *

class AlgoUsingAnalogy(Algo):
    """Abstract class for algos that use an analogy framework"""
    def __init__(self, rm, ur, mr, movieBased=False):
        super().__init__(rm, ur, mr, movieBased)
        self.tuples = [] # list of 3-tuple (for the last prediction only)

    def isSolvable(self, ra, rb, rc):
        """return true if analogical equation is solvable else false"""
        return (ra == rb) or (ra == rc)

    def solve(self, ra, rb, rc):
        """ solve A*(a, b, c, x). Undefined if equation not solvable."""
        return rc - ra + rb

    def tvAStar (self, ra, rb, rc, rd):
        """return the truth value of A*(ra, rb, rc, rd)"""

        # map ratings into [0, 1]
        ra = (ra-1)/4.; rb = (rb-1)/4.; rc = (rc-1)/4.; rd = (rd-1)/4.;
        return min(1 - abs(max(ra, rd) - max(rb, rc)), 1 - abs(min(ra, rd) -
            min(rb, rc)))

    def tvA(self, ra, rb, rc, rd):
        """return the truth value of A(ra, rb, rc, rd)"""

        # map ratings into [0, 1]
        ra = (ra-1)/4.; rb = (rb-1)/4.; rc = (rc-1)/4.; rd = (rd-1)/4.;
        if (ra >= rb and rc >= rd) or (ra <= rb and rc <= rd):
            return 1 - abs((ra-rb) - (rc-rd))
        else:
            return 1 - max(abs(ra-rb), abs(rc-rd))


class AlgoParall(AlgoUsingSim, AlgoUsingAnalogy):
    """geometrical analogy based recommender"""
    def __init__(self, rm, ur, mr, movieBased=False, sim='MSD', k=40,
            **kwargs):
        super().__init__(rm, ur, mr, movieBased, sim)
        self.infos['name'] = 'algoParallKNN' if k else 'algoParall'

        self.k = k

        # if k is specified, look for 3-tuples in the kNN. Else, choose them
        # randomly
        self.gen = self.genkNN if k else self.genRandom

    def genkNN(self, x0, y0):
        """generator to find triplets amongst the kNN"""

        # list of (x, sim(x0, x)) for x having rated m0 or for m rated by x0
        simX0 = [(x, self.simMat[x0, x]) for x in range(1, self.lastXi + 1) if
            self.rm[x, y0] > 0]

        # sort simX0 by similarity decreasing order
        simX0 = sorted(simX0, key=lambda x:x[1], reverse=True)

        # get only the k nearest neighbors
        neighboorsList = [(x, self.rm[x, y0]) for (x, _) in simX0[:self.k]]
        for xa, ra in neighboorsList:
            for xb, rb in neighboorsList:
                for xc, rc in neighboorsList:
                    yield (xa, ra), (xb, rb), (xc, rc)

    def genRandom(self, x0, y0):
        """generator that will return 1000 random triplets"""
        for i in range(1000):
            # randomly choose a, b, and c
            xa, ra = rd.choice(self.yr[y0])
            xb, rb = rd.choice(self.yr[y0])
            xc, rc = rd.choice(self.yr[y0])
            yield (xa, ra), (xb, rb), (xc, rc)


    def estimate(self, u0, m0):
        x0, y0 = self.getx0y0(u0, m0)

        # if there are no ratings for y0, prediction is impossible
        if not self.yr[y0]:
            self.est = 0
            return

        candidates= [] # solutions to analogical equations
        self.tuples = [] # list of 3-tuples that are serve as candidates

        for (xa, ra), (xb, rb), (xc, rc) in self.gen(x0, y0):
            if (xa != xb != xc and xa != xc and
               self.isSolvable(ra, rb, rc)):
                # get info about the abcd 'paralellogram'
                (nYabc0, nrm) = self.getParall(xa, xb, xc, x0)
                if nrm < 1.5 * np.sqrt(nYabc0): # we allow some margin
                    sol = self.solve(ra, rb, rc)
                    candidates.append((sol, nrm, nYabc0))
                    self.tuples.append((xa, xb, xc, nYabc0, sol))

        # if there are candidates, estimate rating as a weighted average
        if candidates:
            ratings = [r for (r, _, _) in candidates]
            norms = [1/(nrm + 1) for (_, nrm, _) in candidates]
            """
            nYs = [nY for (_, _, nY) in candidates]
            self.est = int(round(np.average(ratings, weights=norms)))
            self.est = int(round(np.average(ratings, weights=nYs)))
            """
            self.est = np.average(ratings)
        else:
            self.est = 0


    def getParall(self, xa, xb, xc, x0):
        """return information about the parallelogram formed by xs: number of
        ratings in common and norm of the differences (see formula)"""

        # list of ys that xa, xb, xc, and x0 have commonly rated
        Yabc0 = [y for (y, _) in self.xr[xa] if (self.rm[xb, y] and self.rm[xc, y]
            and self.rm[x0, y])]

        # if there is no common rating
        if not Yabc0:
            return 0, float('inf')

        # lists of ratings for common ys
        xaRs = np.array([self.rm[xa, y] for y in Yabc0])
        xbRs = np.array([self.rm[xb, y] for y in Yabc0])
        xcRs = np.array([self.rm[xc, y] for y in Yabc0])
        x0Rs = np.array([self.rm[x0, y] for y in Yabc0])

        # the closer the norm to zero, the more abcd is in a paralellogram
        # shape
        nrm = np.linalg.norm((xaRs - xbRs) - (xcRs - x0Rs))

        return len(Yabc0), nrm

class AlgoPattern(AlgoUsingAnalogy):
    """analogy based recommender using patterns in 3-tuples"""
    def __init__(self, rm, ur, mr, movieBased=False, **kwargs):
        super().__init__(rm, ur, mr, movieBased)
        self.infos['name'] = 'algoPattern'

    def estimate(self, u0, m0):
        x0, y0 = self.getx0y0(u0, m0)

        # if there are no ratings for y0, prediction is impossible
        if not self.yr[y0]:
            self.est = 0
            return

        candidates= [] # solutions to analogical equations
        self.tuples = [] # list of 3-tuples that are serve as candidates
        tCat1 = np.var([1, 2, 1, 1]) #threshold of variance
        self.tuples = [] # list of 3-tuples that are serve as candidates
        for i in range(1000):
            # randomly choose a, b, and c
            xa, ra = rd.choice(self.yr[y0])
            xb, rb = rd.choice(self.yr[y0])
            xc, rc = rd.choice(self.yr[y0])
            # if pattern is a:a::b:x => swap b and c
            if ra == rb and ra != rc:
                xb, xc = xc, xb
                rb, rc = rc, rb

            cat1 = cat2 = cat3 = 0 # number of 3-tuples belonging to cat1, cat2...
            if xa != xb != xc and xa != xc and self.isSolvable(ra, rb, rc):
                Yabc0 = self.getYabc0(xa, xb, xc, x0)
                if not Yabc0:
                    break
                for y in Yabc0:
                    ray, rby, rcy, r0y = (self.rm[xa, y], self.rm[xb, y],
                        self.rm[xc, y], self.rm[x0, y])
                    # check if 3truple belongs to cat 1
                    # the variance check ensures that ratings are all equal, or
                    # only one differs from the othr with a diff of 1
                    if np.var([ray, rby, rcy, r0y]) <= tCat1:
                        cat1 += 1
                    # check if 3truple belongs to cat 2
                    elif (np.sign(ray-rby) == np.sign(rcy-r0y) and
                        min(abs(ray-rby), abs(rcy-r0y)) <= 2):
                        cat2 += 1

                    # check if 3truple belongs to cat 3
                    else:
                        cat3 += 1

                # Solution filtering depending on pattern
                if ra == rb == rc:
                    if cat1 >= cat2 + cat3:
                        candidates.append(ra)
                        self.tuples.append((xa, xb, xc, len(Yabc0),ra))
                elif abs(ra - rb) >= 2:
                    if cat2 > cat3:
                        candidates.append(rb)
                        self.tuples.append((xa, xb, xc, len(Yabc0),rb))
                else:
                    if cat1 >= cat2 + cat3 or cat2 > cat3:
                        candidates.append(rb)
                        self.tuples.append((xa, xb, xc, len(Yabc0),rb))

        # if there are candidates, estimate rating as a weighted average
        if candidates:
            ratings = [r for r in candidates]
            self.est = np.average(ratings)
        else:
            self.est = 0


    def getYabc0(self, xa, xb, xc, x0):
        # list of ys that xa, xb, xc, and x0 have commonly rated
        return [y for (y, _) in self.xr[xa] if (self.rm[xb, y] and self.rm[xc,
            y] and self.rm[x0, y])]
