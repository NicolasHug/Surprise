.. _getting_started:

Getting Started
===============


.. _load_builtin_example:

Basic usage
-----------

`Surprise <https://nicolashug.github.io/Surprise/>`_ has a set of built-in
:ref:`algorithms<prediction_algorithms>` and :ref:`datasets <dataset>` for you
to play with. In its simplest form, it takes about four lines of code to
evaluate the performance of an algorithm:

.. literalinclude:: ../../examples/basic_usage.py
    :caption: From file ``examples/basic_usage.py``
    :name: basic_usage.py
    :lines: 9-


If `Surprise <https://nicolashug.github.io/Surprise/>`_ cannot find the
`movielens-100k dataset <http://grouplens.org/datasets/movielens/>`_, it will
offer to download it and will store it under the ``.surprise_data`` folder in
your home directory.  The :meth:`split()
<surprise.dataset.DatasetAutoFolds.split>` method automatically splits the
dataset into 3 folds and the :func:`evaluate() <surprise.evaluate.evaluate>`
function runs the cross-validation procedure and compute some :mod:`accuracy
<surprise.accuracy>` measures.


.. _load_custom:

Load a custom dataset
---------------------

You can of course use a custom dataset. `Surprise
<https://nicolashug.github.io/Surprise/>`_ offers two ways of loading a custom
dataset:

- you can either specify a single file (e.g. a csv file) or a pandas dataframe
  with all the ratings and use the :meth:`split
  ()<surprise.dataset.DatasetAutoFolds.split>` method to perform
  cross-validation, or :ref:`train on the whole dataset
  <train_on_whole_trainset>` ;
- or if your dataset is already split into predefined folds, you can specify a
  list of files for training and testing.

Either way, you will need to define a :class:`Reader <surprise.dataset.Reader>`
object for `Surprise <https://nicolashug.github.io/Surprise/>`_ to be able to
parse the file(s) or the dataframe. We'll see now how to handle both cases.

.. _load_from_file_example:

Load an entire dataset from a file or a dataframe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- To load a dataset from a file (e.g. a csv file), you will need the
  :meth:`load_from_file() <surprise.dataset.Dataset.load_from_file>` method:

  .. literalinclude:: ../../examples/load_custom_dataset.py
      :caption: From file ``examples/load_custom_dataset.py``
      :name: load_custom_dataset.py
      :lines: 17-26

  For more details about readers and how to use them, see the :class:`Reader
  class <surprise.dataset.Reader>` documentation.

  .. note::
      As you already know from the previous section, the Movielens-100k dataset
      is built-in so a much quicker way to load the dataset is to do ``data =
      Dataset.load_builtin('ml-100k')``. We will of course ignore this here.

.. _load_from_df_example:

- To load a dataset from a pandas dataframe, you will need the
  :meth:`load_from_df() <surprise.dataset.Dataset.load_from_df>` method. You
  will also need a :class:`Reader<surprise.dataset.Reader>` object, but only
  the ``rating_scale`` parameter must be specified. The dataframe must have
  three columns, corresponding to the user (raw) ids, the item (raw) ids, and
  the ratings in this order. Each row thus corresponds to a given rating. This
  is not restrictive as you can reorder the columns of your dataframe easily.

  .. literalinclude:: ../../examples/load_from_dataframe.py
      :caption: From file ``examples/load_from_dataframe.py``
      :name: load_dom_dataframe.py
      :lines: 19-28

  The dataframe initially looks like this:

  .. parsed-literal::

            itemID  rating    userID
      0       1       3         9
      1       1       2        32
      2       1       4         2
      3       2       3        45
      4       2       1  user_foo


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

We will here get a little deeper on what can `Surprise
<https://nicolashug.github.io/Surprise/>`_ do for you.

.. _tuning_algorithm_parameters:

Tune algorithm parameters with GridSearch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :func:`evaluate() <surprise.evaluate.evaluate>` function gives us the
results on one set of parameters given to the algorithm. If the user wants
to try the algorithm on a different set of parameters, the
:class:`GridSearch <surprise.evaluate.GridSearch>` class comes to the rescue.
Given a ``dict`` of parameters, this
class exhaustively tries all the combination of parameters and helps get the
best combination for an accuracy measurement. It is analogous to
`GridSearchCV <http://scikit-learn.org/stable/modules/generated/sklearn.model
_selection.GridSearchCV.html>`_ from scikit-learn.

For instance, suppose that we want to tune the parameters of the
:class:`SVD <surprise.prediction_algorithms.matrix_factorization.SVD>`. Some of
the parameters of this algorithm are ``n_epochs``, ``lr_all`` and ``reg_all``.
Thus we define a parameters grid as follows

.. literalinclude:: ../../examples/grid_search_usage.py
    :caption: From file ``examples/grid_search_usage.py``
    :name: grid_search_usage.py
    :lines: 13-14

Next we define a :class:`GridSearch <surprise.evaluate.GridSearch>` instance
and give it the class
:class:`SVD <surprise.prediction_algorithms.matrix_factorization.SVD>` as an
algorithm, and ``param_grid``. We will compute both the
RMSE and FCP values for all the combination. Thus the following definition:

.. literalinclude:: ../../examples/grid_search_usage.py
    :caption: From file ``examples/grid_search_usage.py``
    :name: grid_search_usage2.py
    :lines: 16

Now that :class:`GridSearch <surprise.evaluate.GridSearch>` instance is ready,
we can evaluate the algorithm on any data with the
:meth:`GridSearch.evaluate()<surprise.evaluate.GridSearch.evaluate>` method,
exactly like with the regular
:func:`evaluate() <surprise.evaluate.evaluate>` function:

.. literalinclude:: ../../examples/grid_search_usage.py
    :caption: From file ``examples/grid_search_usage.py``
    :name: grid_search_usage3.py
    :lines: 19-22

Everything is ready now to read the results. For example, we get the best RMSE
and FCP scores and parameters as follows:

.. literalinclude:: ../../examples/grid_search_usage.py
    :caption: From file ``examples/grid_search_usage.py``
    :name: grid_search_usage4.py
    :lines: 24-38

For further analysis, we can easily read all the results in a pandas
``DataFrame`` as follows:

.. literalinclude:: ../../examples/grid_search_usage.py
    :caption: From file ``examples/grid_search_usage.py``
    :name: grid_search_usage5.py
    :lines: 40-

.. _iterate_over_folds:

.. _grid_search_note:
.. note::

    Dictionary parameters such as ``bsl_options`` and ``sim_options`` require
    particular treatment. See usage example below:

    .. parsed-literal::

        param_grid = {'k': [10, 20],
                      'sim_options': {'name': ['msd', 'cosine'],
                                      'min_support': [1, 5],
                                      'user_based': [False]}
                      }

    Naturally, both can be combined, for example for the
    :class:`KNNBaseline <surprise.prediction_algorithms.knns.KNNBaseline>`
    algorithm:

    .. parsed-literal::
        param_grid = {'bsl_options': {'method': ['als', 'sgd'],
                                      'reg': [1, 2]},
                      'k': [2, 3],
                      'sim_options': {'name': ['msd', 'cosine'],
                                      'min_support': [1, 5],
                                      'user_based': [False]}
                      }


Manually iterate over folds
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have so far used the :func:`evaluate() <surprise.evaluate.evaluate>`
function that does all the hard work for us. If you want to have better control
on your experiments, you can use the :meth:`folds()
<surprise.dataset.Dataset.folds>` generator of your dataset, and then the
:meth:`train() <surprise.prediction_algorithms.algo_base.AlgoBase.train>` and
:meth:`test() <surprise.prediction_algorithms.algo_base.AlgoBase.test>` methods
of your algorithm on each of the folds:

.. literalinclude:: ../../examples/iterate_over_folds.py
    :caption: From file ``examples/iterate_over_folds.py``
    :name: iterate_over_folds.py
    :lines: 15-

.. _train_on_whole_trainset:

Train on a whole trainset and specifically query for predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will here review how to get a prediction for specified users and items. In
the mean time, we will also review how to train on a whole dataset, without
performing cross-validation (i.e. there is no test set).

The latter is pretty straightforward: all you need is to load a dataset, and
the :meth:`build_full_trainset()
<surprise.dataset.DatasetAutoFolds.build_full_trainset>` method to build the
:class:`trainset <surprise.dataset.Trainset>` and train you algorithm:

.. literalinclude:: ../../examples/query_for_predictions.py
    :caption: From file ``examples/query_for_predictions.py``
    :name: query_for_predictions.py
    :lines: 15-22

Now, there's no way we could call the :meth:`test()
<surprise.prediction_algorithms.algo_base.AlgoBase.test>` method, because we
have no testset. But you can still get predictions for the users and items you
want.

Let's say you're interested in user 196 and item 302 (make sure they're in the
trainset!), and you know that the true rating :math:`r_{ui} = 4`. All you need
is call the :meth:`predict()
<surprise.prediction_algorithms.algo_base.AlgoBase.predict>` method:

.. literalinclude:: ../../examples/query_for_predictions.py
    :caption: From file ``examples/query_for_predictions.py``
    :name: query_for_predictions2.py
    :lines: 28-32

The :meth:`predict()
<surprise.prediction_algorithms.algo_base.AlgoBase.predict>` uses **raw** ids
(read :ref:`this <raw_inner_note>`). As the dataset we have used has been read
from a file, the raw ids are strings (even if they represent numbers).

If the :meth:`predict()
<surprise.prediction_algorithms.algo_base.AlgoBase.predict>` method is called
with user or item ids that were not part of the trainset, it's up to the
algorithm to decide if it still can make a prediction or not. If it can't,
:meth:`predict() <surprise.prediction_algorithms.algo_base.AlgoBase.predict>`
will still predict the mean of all ratings :math:`\mu`.

Obviously, it is perfectly fine to use the :meth:`predict()
<surprise.prediction_algorithms.algo_base.AlgoBase.predict>` method directly
during a cross-validation process. It's then up to you to ensure that the user
and item ids are present in the trainset though.

Command line usage
~~~~~~~~~~~~~~~~~~

Surprise can also be used from the command line, for example:

.. code::

    surprise -algo SVD -params "{'n_epochs': 5, 'verbose': True}" -load-builtin ml-100k -n-folds 3

See detailed usage by running:

.. code::

    surprise -h
