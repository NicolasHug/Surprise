.. _getting_started:

Getting Started
===============


Basic usage
-----------

.. _load_builtin_example:

Pyrec has a set of built-in :ref:`algorithms<prediction_algorithms>` and
:ref:`datasets <dataset>` for you to play with. In its simplest form, it takes
about four lines of code to evaluate the performance of an algorithm:

.. literalinclude:: ../../examples/basic_usage.py
    :caption: examples/basic_usage.py
    :name: basic_usage.py
    :lines: 6-


If Pyrec cannot find the `movielens-100k dataset
<http://grouplens.org/datasets/movielens/>`_, it will offer to download it and
will store it under the ``.pyrec_data`` folder in your home directory.  The
:meth:`split<pyrec.dataset.DatasetAutoFolds.split>` method automatically
splits the dataset into 5 folds and the :func:`evaluate
<pyrec.evaluate.evaluate>` function runs the cross-validation procedure and
compute some :mod:`accuracy <pyrec.accuracy>` measures.



.. _load_from_file_example:
.. _load_from_folds_example:
