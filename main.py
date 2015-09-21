#!/usr/bin/python3
import numpy as np
import random as rd
from collections import defaultdict
import time
import sys

import algo as al
import common as c

if len(sys.argv) != 2:
    sys.exit("Error : Tell me which split to use (1, 2, 3, 4, or 5)")

split = sys.argv[1]

base = open('../ml-100k/u' + split + '.base', 'r')
test = open('../ml-100k/u' + split + '.test', 'r')

rm = np.empty((c.lastMi + 1 , c.lastUi + 1), dtype='int') # the rating matrix
ur = defaultdict(list) # dict of users containing list of (m, rat(u, m))
mr = defaultdict(list) # dict of movies containing list of (u, rat(u, m))

for line in base:
    ui, mi, r, _ = line.split()
    ui = int(ui); mi = int(mi); r = int(r)
    rm[mi, ui] = r
    ur[ui].append((mi, r))
    mr[mi].append((ui, r))

trainStartTime = time.process_time()
#a = al.AlgoRandom(rm, ur, mr)
#a = al.AlgoBasicCollaborative(rm, ur, mr, sim='MSD', movieBased=False)
#a = al.AlgoAnalogy(rm, ur, mr, movieBased=False)
a = al.AlgoParall(rm, ur, mr, movieBased=False)
#a = al.AlgoParallKnn(rm, ur, mr, movieBased=False, sim='MSD')
#a = al.AlgoPattern(rm, ur, mr, movieBased=False)
#a = al.AlgoBaselineOnly(rm, ur, mr, method='opt')
#a = al.AlgoNeighborhoodWithBaseline(rm, ur, mr, movieBased=False, method='opt',sim='MSD')
#a = al.AlgoKNNBelkor(rm, ur, mr, method='opt', movieBased=False)
#a = al.AlgoFactors(rm, ur, mr, movieBased=False)
trainingTime = time.process_time() - trainStartTime

a.infos['params']['split'] = split


rd.seed(0)
testSet = []
for line in test:
    testSet.append(line.split())

smallTestSet = [rd.choice(testSet) for i in range(100)]

testTimeStart = time.process_time()
#for u0, m0, r0, _ in testSet:
for u0, m0, r0, _ in smallTestSet:

    u0 = int(u0); m0 = int(m0); r0 = int(r0)

    print(u0, m0, r0)
    a.estimate(u0, m0)
    a.updatePreds(u0, m0, r0)

    print('-' * 20)

print('-' * 20)
testingTime = time.process_time() - testTimeStart

a.infos['trainingTime'] = trainingTime
a.infos['testingTime'] = testingTime

c.printStats(a.infos['preds'])
a.dumpInfos()
