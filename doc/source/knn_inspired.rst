.. _pred_package_knn_inpired:

k-NN inspired algorithms
------------------------

These are algorithms that are directly derived from a basic nearest neighbors
approach.

.. _actual_k_note:

.. note::

  For each of these algorithms, the actual number of neighbors that are
  aggregated to compute an estimation is necessarily less than or equal to
  :math:`k`. First, there might just not exist enough neighbors and second, the
  sets :math:`N_i^k(u)` and :math:`N_u^k(i)` only include neighbors for which
  the similarity measure is **positive**. It would make no sense to aggregate
  ratings from users (or items) that are negatively correlated. For a given
  prediction, the actual number of neighbors can be retrieved in the
  ``'actual_k'`` field of the ``details`` dictionary of the :class:`prediction
  <surprise.prediction_algorithms.predictions.Prediction>`.

You may want to read the :ref:`User Guide <similarity_measures_configuration>`
on how to configure the ``sim_options`` parameter.

.. autoclass:: surprise.prediction_algorithms.knns.KNNBasic
    :show-inheritance:

.. autoclass:: surprise.prediction_algorithms.knns.KNNWithMeans
    :show-inheritance:

.. autoclass:: surprise.prediction_algorithms.knns.KNNBaseline
    :show-inheritance:
