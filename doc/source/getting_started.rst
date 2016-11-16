.. _getting_started:

Getting Started
===============


.. _load_builtin_example:

Basic usage
-----------

`RecSys <https://github.com/Niourf/RecSys>`_ has a set of built-in
:ref:`algorithms<prediction_algorithms>` and :ref:`datasets <dataset>` for you
to play with. In its simplest form, it takes about four lines of code to
evaluate the performance of an algorithm:

.. literalinclude:: ../../examples/basic_usage.py
    :caption: From file ``examples/basic_usage.py``
    :name: basic_usage.py
    :lines: 9-


If `RecSys <https://github.com/Niourf/RecSys>`_ cannot find the `movielens-100k
dataset <http://grouplens.org/datasets/movielens/>`_, it will offer to download
it and will store it under the ``.recsys_data`` folder in your home directory.
The :meth:`split() <recsys.dataset.DatasetAutoFolds.split>` method
automatically splits the dataset into 3 folds and the :func:`evaluate()
<recsys.evaluate.evaluate>` function runs the cross-validation procedure and
compute some :mod:`accuracy <recsys.accuracy>` measures.


.. _load_custom:

Load a custom dataset
---------------------

You can of course use a custom dataset. `RecSys
<https://github.com/Niourf/RecSys>`_ offers two ways of loading a custom
dataset:

- you can either specify a single file with all the ratings and
  use the :meth:`split ()<recsys.dataset.DatasetAutoFolds.split>` method to
  perform cross-validation ;
- or if your dataset is already split into predefined folds, you can specify a
  list of files for training and testing.

Either way, you will need to define a :class:`Reader <recsys.dataset.Reader>`
object for `RecSys <https://github.com/Niourf/RecSys>`_ to be able to parse the
file(s).

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
    Actually, as the Movielens-100k dataset is builtin, `RecSys
    <https://github.com/Niourf/RecSys>`_ provides with a proper reader so in
    this case, we could have just created the reader like this: ::

      reader = Reader('ml-100k')

For more details about readers and how to use them, see the :class:`Reader
class <recsys.dataset.Reader>` documentation.

.. _load_from_folds_example:

Load a dataset with predefined folds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/load_custom_dataset_predefined_folds.py
    :caption: From file ``examples/load_custom_dataset_predefined_folds.py``
    :name: load_custom_dataset_predefined_folds.py
    :lines: 18-30

Of course, nothing prevents you from only loading a single file for training
and a single file for testing. However, the ``folds_files`` parameter still
needs to be a ``list``.


Advanced usage
--------------

We will here get a little deeper on what can `RecSys
<https://github.com/Niourf/RecSys>`_ do for you.

.. _iterate_over_folds:

Manually iterate over folds
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have so far used the :func:`evaluate() <recsys.evaluate.evaluate>` function
that does all the hard work for us. If you want to have better control on your
experiments, you can use the :meth:`folds() <recsys.dataset.Dataset.folds>`
generator of your dataset, and then the :meth:`train()
<recsys.prediction_algorithms.algo_base.AlgoBase.train>` and :meth:`test()
<recsys.prediction_algorithms.algo_base.AlgoBase.test>` methods of your
algorithm on each of the folds:

.. literalinclude:: ../../examples/iterate_over_folds.py
    :caption: From file ``examples/iterate_over_folds.py``
    :name: iterate_over_folds.py
    :lines: 15-

.. _train_on_whole_trainset:

Train on a whole trainset and specifically query for predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will here review how to get a prediction for specified users and items. In
the mean time, we will also review how to train on a whole dataset, whithout
performing cross-validation (i.e. there is no test set).

The latter is pretty straightforward: all you need is to load a dataset, and
the :meth:`build_full_trainset()
<recsys.dataset.DatasetAutoFolds.build_full_trainset>` method to build the
:class:`trainset <recsys.dataset.Trainset>` and train you algorithm:

.. literalinclude:: ../../examples/query_for_predictions.py
    :caption: From file ``examples/query_for_predictions.py``
    :name: query_for_predictions.py
    :lines: 15-22

Now, there's no way we could call the :meth:`test()
<recsys.prediction_algorithms.algo_base.AlgoBase.test>` method, because we have
no testset. But you can still get predictions for the users and items you want.

Let's say you're interested in user 196 and item 302 (make sure they're in the
trainset!), and you know that the true rating :math:`r_{ui} = 4`. All you need
is call the :meth:`predict()
<recsys.prediction_algorithms.algo_base.AlgoBase.predict>` method:

.. literalinclude:: ../../examples/query_for_predictions.py
    :caption: From file ``examples/query_for_predictions.py``
    :name: query_for_predictions2.py
    :lines: 28-32

If the :meth:`predict()
<recsys.prediction_algorithms.algo_base.AlgoBase.predict>` method is called
with user or item ids that were not part of the trainset, it's up to the
algorithm to decide if he still can make a prediction or not. If it can't,
:meth:`predict() <recsys.prediction_algorithms.algo_base.AlgoBase.predict>`
will still predict the mean of all ratings :math:`\mu`.

.. _raw_inner_note:
.. note::
  Raw ids are ids as defined in a rating file. They can be strings or whatever.
  On trainset creation, each raw id is mapped to a (unique) integer called
  inner id, which is a lot more suitable for `RecSys
  <https://github.com/Niourf/RecSys>`_ to manipulate. To convert a raw id to an
  inner id, you can use the  :meth:`to_inner_uid()
  <recsys.dataset.Trainset.to_inner_uid>` and :meth:`to_inner_iid()
  <recsys.dataset.Trainset.to_inner_iid>` methods of the :class:`trainset
  <recsys.dataset.Trainset>`.

Obviously, it is perfectly fine to use the :meth:`predict()
<recsys.prediction_algorithms.algo_base.AlgoBase.predict>` method directly
during a cross-validation process. It's then up to you to ensure that the user
and item ids are present in the trainset though.

.. _dumping:

Dump the predictions for later analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You may want to save your algorithm predictions along with all the usefull
information about the algorithm. This way, you can run your algorithm once,
save the results, and go back to them whenever you want to inspect in greater
details each of the predictions, and get a good insight on why your algorithm
performs well (or bad!). `RecSys <https://github.com/Niourf/RecSys>`_ provides
with some tools to do that.

You can dump your algorithm predictions either using the :func:`evaluate()
<recsys.evaluate.evaluate>` function, or do it manually with the :func:`dump
<recsys.dump.dump>` function. Either way, an example is worth a thousand words,
so here a few `jupyter <http://jupyter.org/>`_ notebooks:

  - `Dumping and analysis of the KNNBasic algorithm
    <http://nbviewer.jupyter.org/github/Niourf/RecSys/tree/master/examples/notebooks/KNNBasic_analysis.ipynb/>`_.
  - `Comparison of two algorithms
    <http://nbviewer.jupyter.org/github/Niourf/RecSys/tree/master/examples/notebooks/Compare.ipynb/>`_.
