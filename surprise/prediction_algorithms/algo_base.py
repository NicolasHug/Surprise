"""
The :mod:`surprise.prediction_algorithms.algo_base` module defines the base
class :class:`AlgoBase` from which every single prediction algorithm has to
inherit.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .. import similarities as sims
from .predictions import PredictionImpossible
from .predictions import Prediction
from .optimize_baselines import baseline_als
from .optimize_baselines import baseline_sgd


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
            trainset(:obj:`Trainset <surprise.dataset.Trainset>`) : A training
                set, as returned by the :meth:`folds
                <surprise.dataset.Dataset.folds>` method.
        """

        self.trainset = trainset

        # (re) Initialise baselines
        self.bu = self.bi = None

    def predict(self, uid, iid, r_ui, clip=True, verbose=False):
        """Compute the rating prediction for given user and item.

        The ``predict`` method converts raw ids to inner ids and then calls the
        ``estimate`` method which is defined in every derived class. If the
        prediction is impossible (for whatever reason), the prediction is set
        to the global mean of all ratings.

        Args:
            uid: (Raw) id of the user. See :ref:`this note<raw_inner_note>`.
            iid: (Raw) id of the item. See :ref:`this note<raw_inner_note>`.
            r_ui(float): The true rating :math:`r_{ui}`.
            clip(bool): Whether to clip the estimation into the rating scale.
                For example, if :math:`\\hat{r}_{ui}` is :math:`5.5` while the
                rating scale is :math:`[1, 5]`, then :math:`\\hat{r}_{ui}` is
                set to :math:`5`. Same goes if :math:`\\hat{r}_{ui} < 1`.
                Default is ``True``.
            verbose(bool): Whether to print details of the prediction.  Default
                is False.

        Returns:
            A :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>` object.
        """

        # Convert raw ids to inner ids
        try:
            iuid = self.trainset.to_inner_uid(uid)
        except ValueError:
            iuid = 'UKN__' + str(uid)
        try:
            iiid = self.trainset.to_inner_iid(iid)
        except ValueError:
            iiid = 'UKN__' + str(iid)

        details = {}
        try:
            est = self.estimate(iuid, iiid)

            # If the details dict was also returned
            if isinstance(est, tuple):
                est, details = est

            details['was_impossible'] = False

        except PredictionImpossible as e:
            est = self.trainset.global_mean
            details['was_impossible'] = True
            details['reason'] = str(e)

        # Remap the rating into its initial rating scale (because the rating
        # scale was translated so that ratings are all >= 1)
        est -= self.trainset.offset

        # clip estimate into [lower_bound, higher_bound]
        if clip:
            lower_bound, higher_bound = self.trainset.rating_scale
            est = min(higher_bound, est)
            est = max(lower_bound, est)

        pred = Prediction(uid, iid, r_ui, est, details)

        if verbose:
            print(pred)

        return pred

    def test(self, testset, verbose=False):
        """Test the algorithm on given testset.

        Args:
            testset: A test set, as returned by the :meth:`folds
                <surprise.dataset.Dataset.folds>` method.
            verbose(bool): Whether to print details for each predictions.
                Default is False.

        Returns:
            A list of :class:`Prediction\
                <surprise.prediction_algorithms.predictions.Prediction>`
                objects.
        """

        # The ratings are translated back to their original scale.
        predictions = [self.predict(uid,
                                    iid,
                                    r_ui_trans - self.trainset.offset,
                                    verbose=verbose)
                       for (uid, iid, r_ui_trans) in testset]
        return predictions

    def compute_baselines(self):
        """Compute users and items baselines.

        The way baselines are computed depends on the ``bsl_options`` parameter
        passed at the creation of the algoritihm (see
        :ref:`baseline_estimates_configuration`).

        Returns:
            A tuple ``(bu, bi)``, which are users and items baselines."""

        # Firt of, if this method has already been called before on the same
        # trainset, then just return. Indeed, compute_baselines may be called
        # more than one time, for example when a similarity metric (e.g.
        # pearson_baseline) uses baseline estimates.
        if self.bu is not None:
            return self.bu, self.bi

        method = dict(als=baseline_als,
                      sgd=baseline_sgd)

        method_name = self.bsl_options.get('method', 'als')

        try:
            print('Estimating biases using', method_name + '...')
            self.bu, self.bi = method[method_name](self)
            return self.bu, self.bi
        except KeyError:
            raise ValueError('Invalid method ' + method_name +
                             ' for baseline computation.' +
                             ' Available methods are als and sgd.')

    def compute_similarities(self):
        """Build the simlarity matrix.

        The way the similarity matric is computed depends on the
        ``sim_options`` parameter passed at the creation of the algorithm (see
        :ref:`similarity_measures_configuration`).

        Returns:
            The similarity matrix."""

        construction_func = {'cosine': sims.cosine,
                             'msd': sims.msd,
                             'pearson': sims.pearson,
                             'pearson_baseline': sims.pearson_baseline}

        if self.sim_options['user_based']:
            n_x, yr = self.trainset.n_users, self.trainset.ir
        else:
            n_x, yr = self.trainset.n_items, self.trainset.ur

        min_support = self.sim_options.get('min_support', 1)

        args = [n_x, yr, min_support]

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
            print('Computing the {0} similarity matrix...'.format(name))
            sim = construction_func[name](*args)
            print('Done computing similarity matrix.')
            return sim
        except KeyError:
            raise NameError('Wrong sim name ' + name + '. Allowed values ' +
                            'are ' + ', '.join(construction_func.keys()) + '.')
