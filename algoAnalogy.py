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


class AlgoParall(AlgoUsingAnalogy):
    """geometrical analogy based recommender"""
    def __init__(self, rm, ur, mr, movieBased=False):
        super().__init__(rm, ur, mr, movieBased)
        self.infos['name'] = 'algoGilles'

    def estimate(self, u0, m0):
        x0, y0 = self.getx0y0(u0, m0)

        # if there are no ratings for y0, prediction is impossible
        if not self.yr[y0]:
            self.est = 0
            return

        candidates= [] # solutions to analogical equations
        self.tuples = [] # list of 3-tuples that are serve as candidates
        for i in range(1000):
            # randomly choose a, b, and c
            xa, ra = rd.choice(self.yr[y0])
            xb, rb = rd.choice(self.yr[y0])
            xc, rc = rd.choice(self.yr[y0])
            if xa != xb != xc and xa != xc and self.isSolvable(ra, rb, rc):
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

class AlgoParallKnn(AlgoUsingSim,AlgoUsingAnalogy):
     """geometrical analogy based recommender using Knn to get analogical proportions instead of all the guys"""
     

     def __init__(self, rm, ur, mr, movieBased=False, sim='MSD'):
         super().__init__(rm, ur, mr, movieBased=movieBased, sim=sim)

         self.k = 40 #number of chosen neighbours

         self.infos['name'] = 'algoGillesKnn'
         self.infos['params']['k'] = self.k


     def estimate(self, u0, m0):
         x0, y0 = self.getx0y0(u0, m0)
         # list of (x, sim(x0, x)) for x having rated m0 or for m rated by x0
         simX0 = [(x, self.simMat[x0, x]) for x in range(1, self.lastXi + 1) if
             self.rm[x, y0] > 0]

         # if there is nobody to predict the rating, prediction is impossible (=0)
         if not simX0:
             self.est = 0
             return

         # sort simX0 by similarity decreasing order
         simX0 = sorted(simX0, key=lambda x:x[1], reverse=True)

         # get only the Knn guys
         fullList = [x for (x, _) in simX0]
         neighboorsList = [x for (x, _) in simX0[:self.k]]
         #simNeighboors = [sim for (_, sim) in simX0[:self.k]]
         #ratNeighboors = [self.rm[x, y0] for (x, _) in simX0[:self.k]]
                     
         
         candidates= []      # solutions to analogical equations
         #self.tuples = []    # list of 3-tuples that serve as candidates
         # choose a, b, and c among the neighbours here we get a cubic complexity wrt number of neighbours
         seen=[] #to avoid redundancy
         for xa in neighboorsList:
            for xb in neighboorsList:
                 for xc in neighboorsList:
                   if xa != xb != xc and xa != xc and self.isSolvable(self.rm[xa, y0], self.rm[xb, y0], self.rm[xc, y0]):
                 # get info about the abcd 'parallelogram'
                             (nrm,numberOfCommonMovies) = self.getParall(xa, xb, xc, x0)
                             if (nrm < 1.5 * np.sqrt(numberOfCommonMovies)): # we allow some margin
                                 sol = self.solve(self.rm[xa, y0], self.rm[xb, y0], self.rm[xc, y0])
                                 candidates.append((sol, nrm, numberOfCommonMovies))
                                 #seen.append(xa)
                                 #self.tuples.append((xa, xb, xc, nYabc0, sol))

         # if there are candidates, estimate rating as a weighted average
         if candidates:
             ratings = [sol for (sol, _, _) in candidates]
             #norms = [1/(nrm + 1) for (_, nrm, _) in candidates]
             #nYs = [nY for (_, _, nY) in candidates]
             self.est = np.average(ratings)
         else:
             self.est = 0
         print("candidates:",len(candidates),"estim=",self.est)
         
    
     def getParall(self, xa, xb, xc, x0):
         """return all information about the parallelogram formed by xs: number of
         ratings in common and norm of the difference (a-b)-(c-d) (see formula)"""

         # list of movies that xa, xb, xc, and x0 have commonly rated
         # or list of users having seen xa, xb, xc, and x0
         listOfCommon = [y for (y, _) in self.xr[xa] if (self.rm[xb, y] and self.rm[xc, y]
             and self.rm[x0, y])]
         #tv = [] # vector of componentwise truth value
         # if there is no common things
         if not listOfCommon:
             return float('inf'), 0

         # lists of ratings for common things y - 4 vectors with same dimension
         xaRs = np.array([self.rm[xa, y] for y in listOfCommon])
         xbRs = np.array([self.rm[xb, y] for y in listOfCommon])
         xcRs = np.array([self.rm[xc, y] for y in listOfCommon])
         x0Rs = np.array([self.rm[x0, y] for y in listOfCommon])

         # the closer the norm to zero, the more abcd looks like a parallelogram
         # norm is important
         nrm = np.linalg.norm((xaRs - xbRs) - (xcRs - x0Rs))
         
         # list of ratings from xa xb xc x0 for the common things
         #Yabc0 = [(self.rm[xa, y], self.rm[xb, y], self.rm[xc, y], self.rm[x0,y]) for y in listOfCommon]
         #compute the truth value componentwise
         #for (ra, rb, rc, rd) in Yabc0:
             #tv.append(self.tvAStar(ra, rb, rc, rd))
           #  tv.append(self.tvA(ra, rb, rc, rd))
             
         return nrm,  len(listOfCommon)

class AlgoPattern(AlgoUsingAnalogy):
    """analogy based recommender using patterns in 3-tuples"""
    def __init__(self, rm, ur, mr, movieBased=False):
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



