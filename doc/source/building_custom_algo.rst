.. _building_custom_algo:

How to build you own prediction algorithm
=========================================

This page describes how to build a custom prediction algorithm using RecSys.

The basics
~~~~~~~~~~

Want to get your hands dirty? Cool.

Creating your own prediction algorithm is
pretty simple: an algorithm is nothing but a class derived from :class:`AlgoBase
<recsys.prediction_algorithms.bases.AlgoBase>` that has an ``estimate`` method.
This methods takes in a user id, an item id, and returns the estimated rating
:math:`\hat{r}_{ui}`:

.. literalinclude:: ../../examples/building_custom_algorithms/most_basic_algorithm.py
    :caption: From file ``examples/building_custom_algorithms/most_basic_algorithm.py``
    :name: most_basic_algorithm.py
    :lines: 9-

This algorithm is the dumbest we could have thought of: it just predicts a
rating of 3, regardless of users and items.

The ``train`` method
~~~~~~~~~~~~~~~~~~~~

Now, let's make a slightly cleverer algorithm that predicts the average of all
the ratings of the trainset. As this is a constant value that does not depend
on current user or item, we would rather compute it once and for all. This can
be done by defining the ``train`` method:

.. literalinclude:: ../../examples/building_custom_algorithms/most_basic_algorithm2.py
    :caption: From file ``examples/building_custom_algorithms/most_basic_algorithm2.py``
    :name: most_basic_algorithm2.py
    :lines: 15-32


The ``train`` method is called by the :func:`evaluate
<recsys.evaluate.evaluate>` function at each fold of a cross-validation
process, (but you can also :ref:`call it yourself <iterate_over_folds>`).
Before doing anything, you should call the base class :meth:`train
<recsys.prediction_algorithms.bases.AlgoBase.train>` method.

The ``trainset`` attribute
~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the base class :meth:`train
<recsys.prediction_algorithms.bases.AlgoBase.train>` method has returned, all
the info you need about the current training set (rating values, etc...) is
stored in the ``self.trainset`` attribute which is a named tuple with many
fields of interest, for which you have all the details in the :class:`API
Reference <recsys.dataset.Trainset>`.

To illustrate its usage, let's make an algorithm that predicts the mean rating
of the user:

.. literalinclude:: ../../examples/building_custom_algorithms/mean_rating_user.py
    :caption: From file ``examples/building_custom_algorithms/mean_rating_user.py``
    :name: mean_rating_user.py
    :lines: 22-24

Predicting the mean rating for an item would have been done in a similar
fashion using the ``ir`` field. Note that it would have been a better idea to
compute all the user means in the ``train`` method, thus avoiding the same
computations multiple times.


Using similarities and baselines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Should your algorithm use a similarity measure or baseline estimates, you'll
need to accept ``bsl_options`` and ``sim_options`` as parmeters to the
``__init__`` method, and pass them along to the Base class. See how to use
these parameters in the :ref:`prediction_algorithms` section.

Methods :meth:`compute_baselines
<recsys.prediction_algorithms.bases.AlgoBase.compute_baselines>`   and
:meth:`compute_similarities
<recsys.prediction_algorithms.bases.AlgoBase.compute_similarities>` can be
called in the ``train`` method.  Baselines estimates are then available using
the :meth:`get_baseline
<recsys.prediction_algorithms.bases.AlgoBase.get_baseline>` method and
similarities can be retrieved using the ``self.sim`` attribute:

.. literalinclude:: ../../examples/building_custom_algorithms/with_baselines_or_sim.py
    :caption: From file ``examples/building_custom_algorithms/.with_baselines_or_sim.py``
    :name: with_baselines_or_sim.py
    :lines: 14-42


Feel free to explore the prediction_algorithms package `source
<https://github.com/Niourf/RecSys/tree/master/recsys/prediction_algorithms>`_
to get an idea of what can be done.
