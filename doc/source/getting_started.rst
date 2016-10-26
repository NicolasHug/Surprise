.. _getting_started:

Getting Started
===============


.. _load_builtin_example:

Basic usage
-----------

RecSys has a set of built-in :ref:`algorithms<prediction_algorithms>` and
:ref:`datasets <dataset>` for you to play with. In its simplest form, it takes
about four lines of code to evaluate the performance of an algorithm:

.. literalinclude:: ../../examples/basic_usage.py
    :caption: From file ``examples/basic_usage.py``
    :name: basic_usage.py
    :lines: 9-


If RecSys cannot find the `movielens-100k dataset
<http://grouplens.org/datasets/movielens/>`_, it will offer to download it and
will store it under the ``.recsys_data`` folder in your home directory.  The
:meth:`split<recsys.dataset.DatasetAutoFolds.split>` method automatically
splits the dataset into 3 folds and the :func:`evaluate
<recsys.evaluate.evaluate>` function runs the cross-validation procedure and
compute some :mod:`accuracy <recsys.accuracy>` measures.


Load a custom dataset
---------------------

You can of course use a custom dataset. RecSys offers two ways of loading a
custom dataset:

- you can either specify a single file with all the ratings and
  use the :meth:`split<recsys.dataset.DatasetAutoFolds.split>` method to perform
  cross-validation ;
- or if your dataset is already split into predefined folds, you can specify a
  list of files for training and testing.

Either way, you will need to define a :class:`Reader <recsys.dataset.Reader>`
object for RecSys to be able to parse the file(s).

We'll see how to handle both cases with the `movielens-100k dataset
<http://grouplens.org/datasets/movielens/>`_. Of course this is a built-in
dataset, but we will act as if it were not.

.. _load_from_file_example:

Load an entire dataset
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/load_custom_dataset.py
    :caption: From file ``examples/load_custom_dataset.py``
    :name: load_custom_dataset.py
    :lines: 16-25

.. note::
    Actually, as the Movielens-100k dataset is builtin, RecSys provides with a
    proper reader so in this case, we could have just created the reader like
    this: ::

      reader = Reader('ml-100k')

For more details about readers and how to use them, see the :class:`Reader
class <recsys.dataset.Reader>` documentation.

.. _load_from_folds_example:

Load a dataset with predefined folds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/load_custom_dataset_predefined_folds.py
    :caption: From file ``examples/load_custom_dataset_predefined_folds.py``
    :name: load_custom_dataset_predefined_folds.py
    :lines: 17-29

Of course, nothing prevents you from only loading a single file for training
and a single file for testing. However, the ``folds_files`` parameter still
needs to be a ``list`` (or any iterable).


Advanced usage
--------------

.. _iterate_over_folds:

Manually iterate over folds
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have so far used the :func:`evaluate <recsys.evaluate.evaluate>` function
that does all the hard work for us. If you want to have better control on your
experiments, you can use the :meth:`folds <recsys.dataset.Dataset.folds>`
generator of your dataset, and then the :meth:`train
<recsys.prediction_algorithms.bases.AlgoBase.train>` and :meth:`test
<recsys.prediction_algorithms.bases.AlgoBase.test>` methods of your algorithm on
each of the folds:

.. literalinclude:: ../../examples/iterate_over_folds.py
    :caption: From file ``examples/iterate_over_folds.py``
    :name: iterate_over_folds.py
    :lines: 14-

.. _train_on_whole_trainset:

Train on a whole trainset and specifically query for predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO
