"""
The :mod:`recsys.prediction_algorithms.bases` module defines the base class
:class:`AlgoBase` from
which every single prediction algorithm has to inherit.
"""

from collections import defaultdict
from collections import namedtuple
import numpy as np

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})

from .. import similarities as sims
from .. import colors


class PredictionImpossible(Exception):
    """Exception raised when a prediction is impossible."""
    pass

# This is a weird way of creating a named tuple but else the documentation
# would be awful.
class Prediction(namedtuple('Prediction',
                            ['uid', 'iid', 'r0', 'est', 'details'])):
    """A name tuple for storing the results of a prediction.

    Args:
        uid: The user id.
        iid: The item id.
        r0: The true rating :math:`r_{ui}`.
        est: The estimated rating :math:`\\hat{r}_{ui}`.
        details (dict): Stores additional details about the prediction that
            might be useful for later analysis.
        """

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

        # whether the algo will be based on users (basically means that the
        # similarities will be computed between users or between items) if the
        # algo is user based, x denotes a user and y an item if the algo is
        # item based, x denotes an item and y a user
        self.user_based = self.sim_options.get('user_based', True)

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

        if self.user_based:
            self.rm = trainset.rm
            self.xr = trainset.ur
            self.yr = trainset.ir
            self.n_x = trainset.n_users
            self.n_y = trainset.n_items
        else:
            self.rm = defaultdict(int)
            # @TODO: maybe change that...
            for (ui, mi), r in trainset.rm.items():
                self.rm[mi, ui] = r
            self.xr = trainset.ir
            self.yr = trainset.ur
            self.n_x = trainset.n_items
            self.n_y = trainset.n_users

        # number of ratings
        self.n_ratings = len(self.rm)
        # global mean of all ratings
        self.global_mean = np.mean([r for (_, _, r) in self.all_ratings])

        # (re) Initialise baselines and sim structure
        self.x_biases = self.y_biases = None
        self.sim = None

    def predict(self, u0, i0, r0=0, verbose=False):
        """Compute the rating prediction for user u0 and item i0.

        The ``predict`` method calls the ``estimate`` method which is defined
        in every derived class. If the prediction is impossible (for whatever
        reason), the prediction is set to the global mean of all ratings. Also,
        if :math:`\\hat{r}_{ui}` is outside the bounds of the rating scale,
        (e.g. :math:`\\hat{r}_{ui} = 6` for a rating scale of :math:`[1, 5]`),
        then it is capped.

        Args:
            u0: (Inner) id of user.
            i0: (Inner) id of item.
            r0: The true rating :math:`r_{ui}`.
            verbose(bool): Whether to print details of the prediction.  Default
                is False.

        Returns:
            A :obj:`Prediction` object.
        """

        x0, y0 = (u0, i0) if self.user_based else (i0, u0)

        self.pred_details= {}

        try:
            if str(u0).startswith('unknown') or str(i0).startswith('unknown'):
                raise PredictionImpossible('user or item was not part of ' +
                                           'training set')

            est = self.estimate(x0, y0)
            impossible = False
        except PredictionImpossible:
            est = self.global_mean
            impossible = True

        # clip estimate into [self.r_min, self.r_max]
        est = min(self.trainset.r_max, est)
        est = max(self.trainset.r_min, est)

        self.pred_details['was_impossible'] = impossible

        if verbose:
            print('user:', u0, '| item:', i0, '| r0:', r0,
                  '| est: {0:1.2f} | '.format(est), end='')
            err = abs(est - r0)
            col = colors.FAIL if err > 1 else colors.OKGREEN
            print(col + "err = {0:1.2f}".format(err) + colors.ENDC, end=' ')
            if impossible:
                print(colors.FAIL + 'Impossible to predict' + colors.ENDC)
            else:
                print()

        return Prediction(u0, i0, r0, est, self.pred_details)

    def test(self, testset, verbose=False):
        """Test the algorithm on given testset.

        Args:
            testset: A test set, as returned by the :meth:`folds
                <recsys.dataset.Dataset.folds>` method.
            verbose(bool): Whether to print details for each predictions.
                Default is False.

        Returns:
            A list of :obj:`Prediction` objects."""

        predictions = [self.predict(uid, iid, r, verbose=verbose)
                       for (uid, iid, r) in testset]
        return predictions

    def compute_baselines(self):
        """Compute users and items baselines."""

        # Firt of, if this method has already been called before on the same
        # trainset, then just return. Indeed, compute_baselines may be called
        # more than one time, for example when a similarity metric (e.g.
        # pearson_baseline) uses baseline estimates.  I don't quite like the
        # way it's handled but it works...
        if self.x_biases is not None:
            return

        def optimize_sgd():
            """Optimize biases using sgd"""

            reg = self.bsl_options.get('reg', 0.02)
            lr = self.bsl_options.get('learning_rate', 0.005)
            n_epochs = self.bsl_options.get('n_epochs', 20)

            for dummy in range(n_epochs):
                for x, y, r in self.all_ratings:
                    err = (r -
                          (self.global_mean + self.x_biases[x] + self.y_biases[y]))
                    # update x_biases
                    self.x_biases[x] += lr * (err - reg * self.x_biases[x])
                    # udapte y_biases
                    self.y_biases[y] += lr * (err - reg * self.y_biases[y])

        def optimize_als():
            """Alternatively optimize y_biases and x_biases."""

            # This piece of code is largely inspired by that of MyMediaLite:
            # https://github.com/zenogantner/MyMediaLite/blob/master/src/MyMediaLite/RatingPrediction/UserItemBaseline.cs

            reg_u = self.bsl_options.get('reg_u', 15)
            reg_i = self.bsl_options.get('reg_i', 10)
            n_epochs = self.bsl_options.get('n_epochs', 10)

            self.reg_x = reg_u if self.user_based else reg_i
            self.reg_y = reg_i if self.user_based else reg_u

            for dummy in range(n_epochs):
                self.y_biases = np.zeros(self.n_y)
                for y in self.all_ys:
                    devY = sum(r - self.global_mean -
                               self.x_biases[x] for (x, r) in self.yr[y])
                    self.y_biases[y] = devY / (self.reg_y + len(self.yr[y]))

                self.x_biases = np.zeros(self.n_x)
                for x in self.all_xs:
                    devX = sum(r - self.global_mean -
                               self.y_biases[y] for (y, r) in self.xr[x])
                    self.x_biases[x] = devX / (self.reg_x + len(self.xr[x]))

        self.x_biases = np.zeros(self.n_x)
        self.y_biases = np.zeros(self.n_y)

        optimize = dict(als=optimize_als,
                        sgd=optimize_sgd)

        method = self.bsl_options.get('method', 'als')

        try:
            print('Estimating biases using', method + '...')
            optimize[method]()
        except KeyError:
            raise ValueError('invalid method ' + method + ' for baseline ' +
                             'computation. Available methods are als, sgd.')


    def get_baseline(self, x, y):
        return self.global_mean + self.x_biases[x] + self.y_biases[y]

    def compute_similarities(self):
        """Build the simlarity matrix."""

        print("computing the similarity matrix...")
        construction_func = {'cosine' : sims.cosine,
                             'msd' : sims.msd,
                             'pearson' : sims.pearson,
                             'pearson_baseline' : sims.pearson_baseline}

        name = self.sim_options.get('name', 'msd').lower()
        args = [self.n_x, self.yr]
        if name == 'pearson_baseline':
            shrinkage = self.sim_options.get('shrinkage', 100)
            self.compute_baselines()
            args += [self.global_mean, self.x_biases, self.y_biases, shrinkage]

        try:
            self.sim = construction_func[name](*args)
        except KeyError:
            raise NameError('Wrong sim name ' + name + '. Allowed values ' +
                            'are ' + ', '.join(construction_func.keys()) + '.')

    @property
    def all_ratings(self):
        """generator to iter over all ratings"""

        # TODO: why not just use rm.values() ??

        for x, x_ratings in self.xr.items():
            for y, r in x_ratings:
                yield x, y, r

    @property
    def all_xs(self):
        """generator to iter over all xs"""
        return range(self.n_x)

    @property
    def all_ys(self):
        """generator to iter over all ys"""
        return range(self.n_y)
