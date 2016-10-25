.. _pred_package_knn_inpired:

k-NN inspired algorithms
------------------------

These are algorithm that are directly derived from a basic nearest neighbors
approach.

.. _actual_k_note:

.. note::

  For each of these algorithms, the actual number of neighbors that are
  aggregated to compute an estimation is necessarily less than or equal to
  :math:`k`. First, there might just not exist enough neighbors and second, the
  sets :math:`N_i^k(u)` and :math:`N_u^k(i)` only include neighbors for which
  the similarity measure is **positive**. It would make no sense to aggregate
  ratings from users (or items) that are negatively correlated.

.. autoclass:: recsys.prediction_algorithms.knns.KNNBasic
    :show-inheritance:

.. autoclass:: recsys.prediction_algorithms.knns.KNNWithMeans
    :show-inheritance:

.. autoclass:: recsys.prediction_algorithms.knns.KNNBaseline
    :show-inheritance:
