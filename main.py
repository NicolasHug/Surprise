#!/usr/bin/python3
import pandas as pd
import numpy as np
import random as rd
from collections import defaultdict

import algo as al
import common as c

base = open('../ml-100k/u1.base', 'r')
test = open('../ml-100k/u1.test', 'r')

rm = np.empty((c.lastMi + 1 , c.lastUi + 1), dtype='int')
ur = defaultdict(list)
mr = defaultdict(list)

for line in base:
    ui, mi, r, _ = line.split()
    ui = int(ui); mi = int(mi); r = int(r)
    rm[mi, ui] = r
    ur[ui].append((mi, r))
    mr[mi].append((ui, r))

#a = al.AlgoRandom(rm, ur, mr)
#a = al.AlgoBasicCollaborative(rm, ur, mr, movieBased=False)
#a = al.AlgoAnalogy(rm, ur, mr, movieBased=False)
#a = al.AlgoGilles(rm, ur, mr, movieBased=False)
#a = al.AlgoBaselineOnly(rm, ur, mr, method='opt')
a = al.AlgoNeighborhoodWithBaseline(rm, ur, mr, movieBased=False, method='opt')
#a = al.AlgoKNNBelkor(rm, ur, mr, method='opt', movieBased=False)


rd.seed(0)
testSet = []
for line in test:
    testSet.append(line.split())

for _ in range(100):
    u0, m0, r0, _ = rd.choice(testSet)
    """
for u0, m0, r0, _ in testSet:
"""
    u0 = int(u0); m0 = int(m0); r0 = int(r0)

    print(u0, m0, r0)
    a.estimate(u0, m0)
    a.updatePreds(u0, m0, r0)

    print('-' * 20)

print('-' * 20)
c.printStats(a.infos['preds'])
a.dumpInfos()
