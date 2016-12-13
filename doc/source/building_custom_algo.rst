.. _building_custom_algo:

How to build you own prediction algorithm
=========================================

This page describes how to build a custom prediction algorithm using Surprise.

The basics
~~~~~~~~~~

Want to get your hands dirty? Cool.

Creating your own prediction algorithm is pretty simple: an algorithm is
nothing but a class derived from :class:`AlgoBase
<surprise.prediction_algorithms.algo_base.AlgoBase>` that has an ``estimate``
method.  This is the method that is called by the :meth:`predict()
<surprise.prediction_algorithms.algo_base.AlgoBase.predict>` method. It takes
in an **inner** user id, an **inner** item id (see :ref:`this note
<raw_inner_note>`), and returns the estimated rating :math:`\hat{r}_{ui}`:

.. literalinclude:: ../../examples/building_custom_algorithms/most_basic_algorithm.py
    :caption: From file ``examples/building_custom_algorithms/most_basic_algorithm.py``
    :name: most_basic_algorithm.py
    :lines: 9-

This algorithm is the dumbest we could have thought of: it just predicts a
rating of 3, regardless of users and items.

If you want to store additional information about the prediction, you can also
return a dictionary with given details: ::

    def estimate(self, u, i):

        details = {'info1' : 'That was',
                   'info2' : 'easy stuff :)'}
        return 3, details

This dictionary will be stored in the :class:`prediction
<surprise.prediction_algorithms.predictions.Prediction>` as the ``details``
field and can be used :ref:`for later analysis <dumping>`.



The ``train`` method
~~~~~~~~~~~~~~~~~~~~

Now, let's make a slightly cleverer algorithm that predicts the average of all
the ratings of the trainset. As this is a constant value that does not depend
on current user or item, we would rather compute it once and for all. This can
be done by defining the ``train`` method:

.. literalinclude:: ../../examples/building_custom_algorithms/most_basic_algorithm2.py
    :caption: From file ``examples/building_custom_algorithms/most_basic_algorithm2.py``
    :name: most_basic_algorithm2.py
    :lines: 15-35


The ``train`` method is called by the :func:`evaluate
<surprise.evaluate.evaluate>` function at each fold of a cross-validation
process, (but you can also :ref:`call it yourself <iterate_over_folds>`).
Before doing anything, you should call the base class :meth:`train()
<surprise.prediction_algorithms.algo_base.AlgoBase.train>` method.

The ``trainset`` attribute
~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the base class :meth:`train()
<surprise.prediction_algorithms.algo_base.AlgoBase.train>` method has returned,
all the info you need about the current training set (rating values, etc...) is
stored in the ``self.trainset`` attribute. This is a :class:`Trainset
<surprise.dataset.Trainset>` object that has many attributes and methods of
interest for prediction.

To illustrate its usage, let's make an algorithm that predicts an average
between the mean of all ratings, the mean rating of the user and the mean
rating for the item:

.. literalinclude:: ../../examples/building_custom_algorithms/mean_rating_user_item.py
    :caption: From file ``examples/building_custom_algorithms/mean_rating_user_item.py``
    :name: mean_rating_user_item.py
    :lines: 22-35

Note that it would have been a better idea to compute all the user means in the
``train`` method, thus avoiding the same computations multiple times.


When the prediction is impossible
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It's up to your algorithm to decide if it can or cannot yield a prediction. If
the prediction is impossible, then you can raise the
:class:`PredictionImpossible
<surprise.prediction_algorithms.predictions.PredictionImpossible>` exception.
You'll need to import it first): ::

  from surprise import PredictionImpossible


This exception will be caught by the :meth:`predict()
<surprise.prediction_algorithms.algo_base.AlgoBase.predict>` method, and the
estimation :math:`\hat{r}_{ui}` will be set to the global mean of all ratings
:math:`\mu`.

Using similarities and baselines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Should your algorithm use a similarity measure or baseline estimates, you'll
need to accept ``bsl_options`` and ``sim_options`` as parmeters to the
``__init__`` method, and pass them along to the Base class. See how to use
these parameters in the :ref:`prediction_algorithms` section.

Methods :meth:`compute_baselines()
<surprise.prediction_algorithms.algo_base.AlgoBase.compute_baselines>`   and
:meth:`compute_similarities()
<surprise.prediction_algorithms.algo_base.AlgoBase.compute_similarities>` can
be called in the ``train`` method (or anywhere else).

.. literalinclude:: ../../examples/building_custom_algorithms/with_baselines_or_sim.py
    :caption: From file ``examples/building_custom_algorithms/.with_baselines_or_sim.py``
    :name: with_baselines_or_sim.py
    :lines: 15-47


Feel free to explore the prediction_algorithms package `source
<https://github.com/NicolasHug/Surprise/tree/master/surprise/prediction_algorithms>`_
to get an idea of what can be done.
