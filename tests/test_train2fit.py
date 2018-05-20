'''Testing renaming of train() into fit()'''
import pytest

from surprise import AlgoBase
from surprise.model_selection import KFold


def test_new_style_algo(small_ml):
    '''Test that new algorithms (i.e. algoritms that only define fit()) can
    support both calls to fit() and to train()
    - algo.fit() is the new way of doing things
    - supporting algo.train() is needed for the (unlikely?) case where a user
    has defined custom tools that use algo.train().
    '''

    class CustomAlgoFit(AlgoBase):

        def __init__(self):
            AlgoBase.__init__(self)
            self.cnt = -1

        def fit(self, trainset):

            AlgoBase.fit(self, trainset)
            self.est = 3
            self.bu, self.bi = 1, 1
            self.cnt += 1

        def estimate(self, u, i):
            return self.est

    algo = CustomAlgoFit()
    kf = KFold(n_splits=2)
    for i, (trainset, testset) in enumerate(kf.split(small_ml)):
        algo.fit(trainset)
        predictions = algo.test(testset)

        # Make sure AlgoBase.fit has been called
        assert hasattr(algo, 'trainset')
        # Make sure CustomAlgoFit.fit has been called
        assert all(est == 3 for (_, _, _, est, _) in predictions)
        # Make sure AlgoBase.fit is finished before CustomAlgoFit.fit
        assert (algo.bu, algo.bi) == (1, 1)
        # Make sure the rest of fit() is only called once
        assert algo.cnt == i

    algo = CustomAlgoFit()
    for i, (trainset, testset) in enumerate(kf.split(small_ml)):
        with pytest.warns(UserWarning):
            algo.train(trainset)
        predictions = algo.test(testset)

        # Make sure AlgoBase.fit has been called
        assert hasattr(algo, 'trainset')
        # Make sure CustomAlgoFit.fit has been called
        assert all(est == 3 for (_, _, _, est, _) in predictions)
        # Make sure AlgoBase.fit is finished before CustomAlgoFit.fit
        assert (algo.bu, algo.bi) == (1, 1)
        # Make sure the rest of fit() is only called once
        assert algo.cnt == i


def test_old_style_algo(small_ml):
    '''Test that old algorithms (i.e. algoritms that only define train()) can
    support both calls to fit() and to train()
    - supporting algo.fit() is needed so that custom algorithms that only
    define train() can still use up to date tools (such as evalute, which has
    been updated to use fit()).
    - algo.train() is the old way, and must still be supported for custom
    algorithms and tools.
    '''

    class CustomAlgoTrain(AlgoBase):

        def __init__(self):
            AlgoBase.__init__(self)
            self.cnt = -1

        def train(self, trainset):

            AlgoBase.train(self, trainset)
            self.est = 3
            self.bu, self.bi = 1, 1
            self.cnt += 1

        def estimate(self, u, i):
            return self.est

    with pytest.warns(UserWarning):
        algo = CustomAlgoTrain()

    kf = KFold(n_splits=2)
    for i, (trainset, testset) in enumerate(kf.split(small_ml)):
        with pytest.warns(UserWarning):
            algo.fit(trainset)
        predictions = algo.test(testset)

        # Make sure AlgoBase.fit has been called
        assert hasattr(algo, 'trainset')
        # Make sure CustomAlgoFit.train has been called
        assert all(est == 3 for (_, _, _, est, _) in predictions)
        # Make sure AlgoBase.fit is finished before CustomAlgoTrain.train
        assert (algo.bu, algo.bi) == (1, 1)
        # Make sure the rest of train() is only called once
        assert algo.cnt == i

    with pytest.warns(UserWarning):
        algo = CustomAlgoTrain()
    for i, (trainset, testset) in enumerate(kf.split(small_ml)):
        with pytest.warns(UserWarning):
            algo.train(trainset)
        predictions = algo.test(testset)

        # Make sure AlgoBase.fit has been called
        assert hasattr(algo, 'trainset')
        # Make sure CustomAlgoFit.train has been called
        assert all(est == 3 for (_, _, _, est, _) in predictions)
        # Make sure AlgoBase.fit is finished before CustomAlgoTrain.train
        assert (algo.bu, algo.bi) == (1, 1)
        # Make sure the rest of train() is only called once
        assert algo.cnt == i
