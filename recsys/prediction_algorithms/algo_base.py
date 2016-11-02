"""
The :mod:`recsys.prediction_algorithms.bases` module defines the base class
:class:`AlgoBase` from
which every single prediction algorithm has to inherit.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import defaultdict
import numpy as np

from .. import similarities as sims
from .predictions import PredictionImpossible
from .predictions import Prediction



class AlgoBase:
    """Abstract class where is defined the basic behaviour of a prediction
    algorithm.

    Keyword Args:
        baseline_options(dict, optional): If the algorithm needs to compute a
            baseline estimate, the ``baseline_options`` parameter is used to
            configure how they are computed. See
            :ref:`baseline_estimates_configuration` for usage.
    """

    def __init__(self, **kwargs):


        self.bsl_options = kwargs.get('bsl_options', {})
        self.sim_options = kwargs.get('sim_options', {})
        if 'user_based' not in self.sim_options:
            self.sim_options['user_based'] = True

    def train(self, trainset):
        """Train an algorithm on a given training set.

        This method is called by every derived class as the first basic step
        for training an algorithm. It basically just initializes some internal
        structures and set the self.trainset attribute.

        Args:
            trainset(:obj:`Trainset <recsys.dataset.Trainset>`) : A training
                set, as returned by the :meth:`folds
                <recsys.dataset.Dataset.folds>` method.
        """

        self.trainset = trainset

        # (re) Initialise baselines and sim structure
        self.bu = self.bi = None
        self.sim = None

    def predict(self, uid, iid, r=0, verbose=False):
        """Compute the rating prediction for given user and item.

        The ``predict`` method calls the ``estimate`` method which is defined
        in every derived class. If the prediction is impossible (for whatever
        reason), the prediction is set to the global mean of all ratings. Also,
        if :math:`\\hat{r}_{ui}` is outside the bounds of the rating scale,
        (e.g. :math:`\\hat{r}_{ui} = 6` for a rating scale of :math:`[1, 5]`),
        then it is capped.

        Args:
            uid: (Inner) id of user. See :ref:`this note<raw_inner_note>`.
            iid: (Inner) id of item. See :ref:`this note<raw_inner_note>`.
            r: The true rating :math:`r_{ui}`.
            verbose(bool): Whether to print details of the prediction.  Default
                is False.

        Returns:
            A :obj:`Prediction <recsys.prediction_algorithms.predictions.Prediction>` object.
        """

        details = {}

        try:
            est = self.estimate(uid, iid)

            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.trainset.global_mean
            details['was_impossible'] = True
            details['reason'] = str(e)

        # clip estimate into [self.r_min, self.r_max]
        est = min(self.trainset.r_max, est)
        est = max(self.trainset.r_min, est)

        pred = Prediction(uid, iid, r, est, details)

        if verbose:
            print(pred)

        return  pred

    def test(self, testset, verbose=False):
        """Test the algorithm on given testset.

        Args:
            testset: A test set, as returned by the :meth:`folds
                <recsys.dataset.Dataset.folds>` method.
            verbose(bool): Whether to print details for each predictions.
                Default is False.

        Returns:
            A list of :class:`Prediction <recsys.prediction_algorithms.predictions.Prediction>` objects.
        """

        predictions = [self.predict(uid, iid, r, verbose=verbose)
                       for (uid, iid, r) in testset]
        return predictions

    def compute_baselines(self):
        """Compute users and items baselines."""

        # Firt of, if this method has already been called before on the same
        # trainset, then just return. Indeed, compute_baselines may be called
        # more than one time, for example when a similarity metric (e.g.
        # pearson_baseline) uses baseline estimates.
        if self.bu is not None:
            return self.bu, self.bi

        def optimize_sgd():
            """Optimize biases using sgd"""

            bu = np.zeros(self.trainset.n_users)
            bi = np.zeros(self.trainset.n_items)

            reg = self.bsl_options.get('reg', 0.02)
            lr = self.bsl_options.get('learning_rate', 0.005)
            n_epochs = self.bsl_options.get('n_epochs', 20)

            for dummy in range(n_epochs):
                for u, i, r in self.trainset.all_ratings():
                    err = (r - (self.trainset.global_mean + bu[u] + bi[i]))
                    bu[u] += lr * (err - reg * bu[u])
                    bi[i] += lr * (err - reg * bi[i])

            return bu, bi

        def optimize_als():
            """Alternatively optimize user biases and and item biases."""

            # This piece of code is largely inspired by that of MyMediaLite:
            # https://github.com/zenogantner/MyMediaLite/blob/master/src/MyMediaLite/RatingPrediction/UserItemBaseline.cs
            # see also https://www.youtube.com/watch?v=gCaOa3W9kM0&t=32m55s
            # (Alex Smola on RS, ML Class 10-701)

            bu = np.zeros(self.trainset.n_users)
            bi = np.zeros(self.trainset.n_items)

            reg_u = self.bsl_options.get('reg_u', 15)
            reg_i = self.bsl_options.get('reg_i', 10)
            n_epochs = self.bsl_options.get('n_epochs', 10)

            for dummy in range(n_epochs):
                bi = np.zeros(self.trainset.n_items)
                for i in self.trainset.all_items():
                    devI = sum(r - self.trainset.global_mean -
                               bu[u] for (u, r) in self.trainset.ir[i])
                    bi[i] = devI / (reg_i + len(self.trainset.ir[i]))

                bu = np.zeros(self.trainset.n_users)
                for u in self.trainset.all_users():
                    devU = sum(r - self.trainset.global_mean -
                               bi[i] for (i, r) in self.trainset.ur[u])
                    bu[u] = devU / (reg_u + len(self.trainset.ur[u]))

            return bu, bi


        optimize = dict(als=optimize_als,
                        sgd=optimize_sgd)

        method = self.bsl_options.get('method', 'als')

        try:
            print('Estimating biases using', method + '...')
            self.bu, self.bi = optimize[method]()
            return self.bu, self.bi
        except KeyError:
            raise ValueError('invalid method ' + method + ' for baseline ' +
                             'computation. Available methods are als, sgd.')


    def compute_similarities(self):
        """Build the simlarity matrix."""

        print("computing the similarity matrix...")
        construction_func = {'cosine' : sims.cosine,
                             'msd' : sims.msd,
                             'pearson' : sims.pearson,
                             'pearson_baseline' : sims.pearson_baseline}

        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        args = [self.n_x, self.yr]

        name = self.sim_options.get('name', 'msd').lower()
        if name == 'pearson_baseline':
            shrinkage = self.sim_options.get('shrinkage', 100)
            bu, bi = self.compute_baselines()
            if self.sim_options['user_based']:
                bx, by = bu, bi
            else:
                bx, by = bi, bu

            args += [self.trainset.global_mean, bx, by, shrinkage]

        try:
            self.sim = construction_func[name](*args)
        except KeyError:
            raise NameError('Wrong sim name ' + name + '. Allowed values ' +
                            'are ' + ', '.join(construction_func.keys()) + '.')
