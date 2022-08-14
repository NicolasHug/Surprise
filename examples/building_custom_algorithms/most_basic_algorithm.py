"""
This module descibes how to build your own prediction algorithm. Please refer
to User Guide for more insight.
"""


from surprise import AlgoBase, Dataset
from surprise.model_selection import cross_validate


class MyOwnAlgorithm(AlgoBase):
    def __init__(self):

        # Always call base method before doing anything.
        AlgoBase.__init__(self)

    def estimate(self, u, i):

        return 3


data = Dataset.load_builtin("ml-100k")
algo = MyOwnAlgorithm()

cross_validate(algo, data, verbose=True)
