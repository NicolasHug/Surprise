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

If you do not want to configure the way baselines are computed, you don't have
to: the default parameteres will do just fine. However, if you want to have
better control on these parameters, we will need to get dirty and dive into the
ugly details of baseline computation.

A baseline estimate :math:`b_{ui}` (sometimes called *bias*) is in itself a
rating prediction. Its goal is to capture the fact that some users are more
inclined to give good ratings than others, and symmetrically, some items tend
to be rated leniently:

.. math::
    b_{ui} = \mu + b_u + b_i

We define an error :math:`e_{ui}` by:

.. math::
    e_{ui} = r_{ui} - b_{ui}

The way baselines are computed is by minimizing the following cost function:

.. math::
    \sum_{r_{ui} \in R_{train}} e_{ui}^2 + \lambda_4(b_u^2 + b_i^2)

where :math:`\lambda_4` is a regularization parameter. A classical stochastic
gradient descent (SGD) would perform the following steps ``n_epoch`` times:

* :math:`b_u \leftarrow b_u + \gamma (e_{ui} - \lambda_4 b_u)`
* :math:`b_i \leftarrow b_i + \gamma (e_{ui} - \lambda_4 b_i)`

where :math:`\gamma` is the learning rate and all values of :math:`b_u` and
:math:`b_i` are first initialized to zero.


Similarities configuration
--------------------------
