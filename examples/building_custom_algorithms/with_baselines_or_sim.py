"""
This module descibes how to build your own prediction algorithm. Please refer
to User Guide for more insight.
"""


from surprise import AlgoBase, Dataset, PredictionImpossible
from surprise.model_selection import cross_validate


class MyOwnAlgorithm(AlgoBase):
    def __init__(self, sim_options={}, bsl_options={}):

        AlgoBase.__init__(self, sim_options=sim_options, bsl_options=bsl_options)

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        # Compute baselines and similarities
        self.bu, self.bi = self.compute_baselines()
        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible("User and/or item is unknown.")

        # Compute similarities between u and v, where v describes all other
        # users that have also rated item i.
        neighbors = [(v, self.sim[u, v]) for (v, r) in self.trainset.ir[i]]
        # Sort these neighbors by similarity
        neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)

        print("The 3 nearest neighbors of user", str(u), "are:")
        for v, sim_uv in neighbors[:3]:
            print(f"user {v} with sim {sim_uv:1.2f}")

        # ... Aaaaand return the baseline estimate anyway ;)
        bsl = self.trainset.global_mean + self.bu[u] + self.bi[i]
        return bsl


data = Dataset.load_builtin("ml-100k")
algo = MyOwnAlgorithm()

cross_validate(algo, data, verbose=True)
