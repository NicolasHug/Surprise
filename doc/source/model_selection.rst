.. _model_selection:

The model_selection package
---------------------------

Surprise provides various tools to run cross-validation procedures and search
the best parameters for a prediction algorithm. The tools presented here are
all heavily inspired from the excellent `scikit learn
<http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection>`_
library.


.. _cross_validation_iterators_api:

Cross validation iterators
==========================

.. automodule:: surprise.model_selection.split
    :members:
    :exclude-members: get_cv, get_rng

Cross validation
================

.. autofunction:: surprise.model_selection.validation.cross_validate

Parameter search
================

.. autoclass:: surprise.model_selection.search.GridSearchCV
    :members:
    :inherited-members:

.. autoclass:: surprise.model_selection.search.RandomizedSearchCV
    :members:
    :inherited-members:

