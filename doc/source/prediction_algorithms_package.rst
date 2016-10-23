.. _prediction_algorithms_package:

prediction_algorithms package
=============================

.. automodule:: recsys.prediction_algorithms

You may want to check the :ref:`notation_standards` before diving into the
formulas.


The algorithm base class
------------------------

.. automodule:: recsys.prediction_algorithms.bases
    :members:
    :exclude-members: all_ratings, all_xs, all_ys

Basic algorithms
----------------

These are basic algorithm that do not do much work but that are still useful
for comparing accuracies.

.. autoclass:: recsys.prediction_algorithms.random_pred.NormalPredictor
    :show-inheritance:

.. autoclass:: recsys.prediction_algorithms.baseline_only.BaselineOnly
    :show-inheritance:


k-NN inspired algorithms
------------------------

These are algorithm that are directly derived from a basic nearest neighbors
approach.

.. autoclass:: recsys.prediction_algorithms.knns.KNNBasic
    :show-inheritance:

.. autoclass:: recsys.prediction_algorithms.knns.KNNWithMeans
    :show-inheritance:

.. autoclass:: recsys.prediction_algorithms.knns.KNNBaseline
    :show-inheritance:
