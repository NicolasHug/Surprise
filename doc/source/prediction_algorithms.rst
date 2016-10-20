.. _prediction_algorithms:

Prediction algorithms
=====================

Pyrec provides with a bunch of built-in algorithms. You can find the details of
each of these in the :mod:`pyrec.prediction_algorithms` package documentation.

Some of these algorithms may use *baseline estimates*, some may use a
similarity metric. We will here review how to configure the way baselines and
similarities are computed.


Baselines estimates configuration
---------------------------------

.. note::
  If you do not want to configure the way baselines are computed, you don't
  have to: the default parameteres will do just fine.

Before continuing, you may want to read section 2.1 of `Factor in the
Neighbors: Scalable and Accurate Collaborative Filtering
<http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf>`_ by
Yehuda Koren to get a good idea of what are baseline estimates.

Baselines can be computed in two different ways:

* Using Stochastic Gradient Descent (SGD).
* Using Alternating Least Squares (ALS).

You can configure the way baselines are computed using the ``bsl_options``
parameter passed at the creation of an algorithm. This parameter is a
dictionary for which the key ``method`` indicates the method to use. Accepted
values are ``'als'`` (default) and ``'sgd'``. Depending on its value, other
options may be set.

.. math::
    b_{ui} = \mu + b_u + b_i

We define an error :math:`e_{ui}` by:

.. math::
    e_{ui} = r_{ui} - b_{ui}

The way baselines are computed is by minimizing the following cost function:

.. math::
    \sum_{r_{ui} \in R_{train}} e_{ui}^2 + \lambda(b_u^2 + b_i^2)

where :math:`\lambda` is a regularization parameter. A classical SGD would
perform the following steps ``n_epoch`` times:

* :math:`b_u \leftarrow b_u + \gamma (e_{ui} - \lambda_4 b_u)`
* :math:`b_i \leftarrow b_i + \gamma (e_{ui} - \lambda_4 b_i)`

where :math:`\gamma` is the learning rate and all values of :math:`b_u` and
:math:`b_i` are first initialized to zero.


Similarities configuration
--------------------------
