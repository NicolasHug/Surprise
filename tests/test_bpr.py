from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
from collections import defaultdict

import numpy as np

from surprise import BPRMF
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut


data_file = os.path.join(os.path.dirname(__file__), './u1_ml100k_train')
data = Dataset.load_from_file(data_file, Reader('ml-100k'))

data = Dataset.load_builtin('ml-100k')
data.binarize(threshold=0)


def AUC(pred_testset, pred_anti_testset, trainset):

    pred_u_anti_testset = defaultdict(list)
    for u, j, r_uj, hat_x_uj, _ in pred_anti_testset:
        pred_u_anti_testset[u].append(hat_x_uj)

    AUC = 0
    for u, i, r_ui, hat_x_ui, _ in pred_testset:
        AUC_u = 0
        for hat_x_uj in pred_u_anti_testset[u]:
            AUC_u += (hat_x_ui > hat_x_uj)
        if pred_u_anti_testset[u]:
            AUC_u /= len(pred_u_anti_testset[u])

        AUC += AUC_u
    AUC /= trainset.n_users

    return AUC

from surprise import AlgoBase

class Dumb(AlgoBase):
    def estimate(self, u, i):
        return np.random.rand()

def test_zob():

    mf = BPRMF(n_epochs=100000)
    mf = Dumb()
    loo = LeaveOneOut()
    for trainset, testset in loo.split(data):

        mf.fit(trainset)
        print('done fitting')
        pred_testset = mf.test(testset, verbose=False)
        pred_anti_testset = mf.test(trainset.build_anti_testset(fill=0),
                                    verbose=False)
        print(np.mean([e for (_, _, _, e, _) in pred_testset]))
        print(np.mean([e for (_, _, _, e, _) in pred_anti_testset]))
        print(AUC(pred_testset, pred_anti_testset, trainset))
        print()
