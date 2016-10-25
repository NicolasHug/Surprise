.. _prediction_algorithms:

Prediction algorithms
=====================

RecSys provides with a bunch of built-in algorithms. You can find the details
of each of these in the :mod:`recsys.prediction_algorithms` package
documentation.

Every algorithm is part of the global RecSys namespace, so you only need to
import their names from the RecSys package, for example: ::

    from recsys import KNNBasic
    algo = KNNBasic()


Some of these algorithms may use :ref:`baseline estimates
<baseline_estimates_configuration>`, some may use a :ref:`similarity measure
<similarity_measures_configuration>`. We will here review how to configure the
way baselines and similarities are computed.


.. _baseline_estimates_configuration:

Baselines estimates configuration
---------------------------------

.. note::
  If you do not want to configure the way baselines are computed, you don't
  have to: the default parameteres will do just fine.

Before continuing, you may want to read section 2.1 of `Factor in the
Neighbors: Scalable and Accurate Collaborative Filtering
<http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf>`_ by
Yehuda Koren to get a good idea of what are baseline estimates.

Baselines can be estimated in two different ways:

* Using Stochastic Gradient Descent (SGD).
* Using Alternating Least Squares (ALS).

You can configure the way baselines are computed using the ``bsl_options``
parameter passed at the creation of an algorithm. This parameter is a
dictionary for which the key ``'method'`` indicates the method to use. Accepted
values are ``'als'`` (default) and ``'sgd'``. Depending on its value, other
options may be set. For ALS:

- ``'reg_i'``: The regularization parameter for items. Corresponding to
  :math:`\lambda_2` in the `paper
  <http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf>`_.
  Default is 10.
- ``'reg_u'``: The regularization parameter for users, orresponding to
  :math:`\lambda_3` in the `paper
  <http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf>`_.
  Default is 15.
- ``'n_epochs'``: The number of iteration of the ALS procedure. Default is 10.
  Note that in the `paper
  <http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf>`_, what
  is described is a **single** iteration ALS process.

And for SGD:

- ``'reg'``: The regularization parameter of the cost function that is
  optimized, corresponding to :math:`\lambda_1` and then :math:`\lambda_5` in
  the `paper
  <http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf>`_.
  Default is 0.02.
- ``'learning_rate'``: The learning rate of SGD, corresponding to
  :math:`\gamma` in the `paper
  <http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf>`_.
  Default is 0.005.
- ``'n_epochs'``: The number of iteration of the SGD procedure. Default is 20. 

.. note::
  For both procedures (ALS and SGD), user and item biases (:math:`b_u` and
  :math:`b_i`) are initialized to zero.

Usage examples:

.. literalinclude:: ../../examples/baselines_conf.py
    :caption: From file ``examples/baselines_conf.py``
    :name: baselines_als
    :lines: 19-25

.. literalinclude:: ../../examples/baselines_conf.py
    :caption: From file ``examples/baselines_conf.py``
    :name: baselines_sgd
    :lines: 30-34

Note that some similarity measures may use baselines, such as the
:func:`pearson_baseline <recsys.similarities.pearson_baseline>` similarity.
Configuration works just the same, whether the baselines are used in the actual
prediction :math:`\hat{r}_{ui}` or not:

.. literalinclude:: ../../examples/baselines_conf.py
    :caption: From file ``examples/baselines_conf.py``
    :name: baselines_als_pearson_sim
    :lines: 40-44


This leads us to similarity measure configuration, which we will review right
now.

.. _similarity_measures_configuration:

Similarity measure configuration
--------------------------------

Many algorithms use a similarity measure to estimate a rating. The way they can
be configured is done in a similar fashion as for baseline ratings: you just
need to pass a ``sim_options`` argument at the creation of an algorithm. This
argument is a dictionary with the following (all optional) keys:

- ``'name'``: The name of the similarity to use, as defined in the
  :mod:`similarities <recsys.similarities>` module. Default is ``'MSD'``.
- ``'user_based'``: Whether similarities will be computed between users or
  between items. This has a **huge** impact on the performance of a prediction
  algorithm.  Default is ``True``.
- ``'shrinkage'``: Shrinkage parameter to apply (only relevent for
  :func:`pearson_baseline <recsys.similarities.pearson_baseline>` similarity).
  Default is 100.

Usage examples:

.. literalinclude:: ../../examples/similarity_conf.py
    :caption: From file ``examples/similarity_conf.py``
    :name: sim_conf_cos
    :lines: 18-20

.. literalinclude:: ../../examples/similarity_conf.py
    :caption: From file ``examples/similarity_conf.py``
    :name: sim_conf_pearson_baseline
    :lines: 25-27

.. seealso::
    The :mod:`similarities <recsys.similarities>` module.
