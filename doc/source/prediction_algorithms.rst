.. _prediction_algorithms:

Using prediction algorithms
===========================

Surprise provides a bunch of built-in algorithms. All algorithms derive from
the :class:`AlgoBase <surprise.prediction_algorithms.algo_base.AlgoBase>` base
class, where are implemented some key methods (e.g. :meth:`predict
<surprise.prediction_algorithms.algo_base.AlgoBase.predict>`, :meth:`fit
<surprise.prediction_algorithms.algo_base.AlgoBase.fit>` and :meth:`test
<surprise.prediction_algorithms.algo_base.AlgoBase.test>`). The list and
details of the available prediction algorithms can be found in the
:mod:`prediction_algorithms <surprise.prediction_algorithms>` package
documentation.

Every algorithm is part of the global Surprise namespace, so you only need to
import their names from the Surprise package, for example: ::

    from surprise import KNNBasic
    algo = KNNBasic()


Some of these algorithms may use :ref:`baseline estimates
<baseline_estimates_configuration>`, some may use a :ref:`similarity measure
<similarity_measures_configuration>`. We will here review how to configure the
way baselines and similarities are computed.


.. _baseline_estimates_configuration:

Baselines estimates configuration
---------------------------------


.. note::
  This section only applies to algorithms (or similarity measures) that try to
  minimize the following regularized squared error (or equivalent):

  .. math::
    \sum_{r_{ui} \in R_{train}} \left(r_{ui} - (\mu + b_u + b_i)\right)^2 +
    \lambda \left(b_u^2 + b_i^2 \right).

  For algorithms using baselines in another objective function (e.g. the
  :class:`SVD <surprise.prediction_algorithms.matrix_factorization.SVD>`
  algorithm), the baseline configuration is done differently and is specific to
  each algorithm. Please refer to their own documentation.

First of all, if you do not want to configure the way baselines are computed,
you don't have to: the default parameters will do just fine. If you do want to
well... This is for you.

You may want to read section 2.1 of :cite:`Koren:2010` to get a good idea of
what are baseline estimates.

Baselines can be estimated in two different ways:

* Using Stochastic Gradient Descent (SGD).
* Using Alternating Least Squares (ALS).

You can configure the way baselines are computed using the ``bsl_options``
parameter passed at the creation of an algorithm. This parameter is a
dictionary for which the key ``'method'`` indicates the method to use. Accepted
values are ``'als'`` (default) and ``'sgd'``. Depending on its value, other
options may be set. For ALS:

- ``'reg_i'``: The regularization parameter for items. Corresponding to
  :math:`\lambda_2` in :cite:`Koren:2010`.  Default is ``10``.
- ``'reg_u'``: The regularization parameter for users. Corresponding to
  :math:`\lambda_3` in :cite:`Koren:2010`.  Default is ``15``.
- ``'n_epochs'``: The number of iteration of the ALS procedure. Default is
  ``10``.  Note that in :cite:`Koren:2010`, what is described is a **single**
  iteration ALS process.

And for SGD:

- ``'reg'``: The regularization parameter of the cost function that is
  optimized, corresponding to :math:`\lambda_1` in
  :cite:`Koren:2010`. Default is ``0.02``.
- ``'learning_rate'``: The learning rate of SGD, corresponding to
  :math:`\gamma` in :cite:`Koren:2010`.  Default is ``0.005``.
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
:func:`pearson_baseline <surprise.similarities.pearson_baseline>` similarity.
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
  :mod:`similarities <surprise.similarities>` module. Default is ``'MSD'``.
- ``'user_based'``: Whether similarities will be computed between users or
  between items. This has a **huge** impact on the performance of a prediction
  algorithm.  Default is ``True``.
- ``'min_support'``: The minimum number of common items (when ``'user_based'``
  is ``'True'``) or minimum number of common users (when ``'user_based'`` is
  ``'False'``) for the similarity not to be zero. Simply put, if
  :math:`|I_{uv}| < \text{min_support}` then :math:`\text{sim}(u, v) = 0`. The
  same goes for items.
- ``'shrinkage'``: Shrinkage parameter to apply (only relevant for
  :func:`pearson_baseline <surprise.similarities.pearson_baseline>` similarity).
  Default is 100.

Usage examples:

.. literalinclude:: ../../examples/similarity_conf.py
    :caption: From file ``examples/similarity_conf.py``
    :name: sim_conf_cos
    :lines: 18-21

.. literalinclude:: ../../examples/similarity_conf.py
    :caption: From file ``examples/similarity_conf.py``
    :name: sim_conf_pearson_baseline
    :lines: 26-29

.. seealso::
    The :mod:`similarities <surprise.similarities>` module.
