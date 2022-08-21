.. _getting_started:

Getting Started
===============


Basic usage
-----------

.. _cross_validate_example:

Automatic cross-validation
~~~~~~~~~~~~~~~~~~~~~~~~~~

`Surprise <https://nicolashug.github.io/Surprise/>`_ has a set of built-in
:ref:`algorithms<prediction_algorithms>` and :ref:`datasets <dataset>` for you
to play with. In its simplest form, it only takes a few lines of code to
run a cross-validation procedure:

.. literalinclude:: ../../examples/basic_usage.py
    :caption: From file ``examples/basic_usage.py``
    :name: basic_usage.py
    :lines: 7-

The result should be as follows (actual values may vary due to randomization):

.. parsed-literal::

    Evaluating RMSE, MAE of algorithm SVD on 5 split(s).

                Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std
    RMSE        0.9311  0.9370  0.9320  0.9317  0.9391  0.9342  0.0032
    MAE         0.7350  0.7375  0.7341  0.7342  0.7375  0.7357  0.0015
    Fit time    6.53    7.11    7.23    7.15    3.99    6.40    1.23
    Test time   0.26    0.26    0.25    0.15    0.13    0.21    0.06


The :meth:`load_builtin() <surprise.dataset.Dataset.load_builtin>` method will
offer to download the `movielens-100k dataset
<https://grouplens.org/datasets/movielens/>`_ if it has not already been
downloaded, and it will save it in the ``.surprise_data`` folder in your home
directory (you can also choose to save it :ref:`somewhere else <data_folder>`).

We are here using the well-known
:class:`SVD<surprise.prediction_algorithms.matrix_factorization.SVD>`
algorithm, but many other algorithms are available. See
:ref:`prediction_algorithms` for more details.

The :func:`cross_validate()<surprise.model_selection.validation.cross_validate>`
function runs a cross-validation procedure according to the ``cv`` argument,
and computes some :mod:`accuracy <surprise.accuracy>` measures. We are here
using a classical 5-fold cross-validation, but fancier iterators can be used
(see :ref:`here <cross_validation_iterators_api>`).

Train-test split and the fit() method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _train_test_split_example:

If you don't want to run a full cross-validation procedure, you can use the
:func:`train_test_split() <surprise.model_selection.split.train_test_split>`
to sample a trainset and a testset with given sizes, and use the :mod:`accuracy
metric<surprise.accuracy>` of your chosing. You'll need to use the :meth:`fit()
<surprise.prediction_algorithms.algo_base.AlgoBase.fit>` method which will
train the algorithm on the trainset, and the :meth:`test()
<surprise.prediction_algorithms.algo_base.AlgoBase.test>` method which will
return the predictions made from the testset:

.. literalinclude:: ../../examples/train_test_split.py
    :caption: From file ``examples/train_test_split.py``
    :name: train_test_split.py
    :lines: 6-

Result:

.. parsed-literal::

    RMSE: 0.9411

Note that you can train and test an algorithm with the following one-line:

.. parsed-literal::

    predictions = algo.fit(trainset).test(testset)


In some cases, your trainset and testset are already defined by some files.
Please refer to :ref:`this section <load_from_folds_example>` to handle such cases.


.. _train_on_whole_trainset:

Train on a whole trainset and the predict() method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Obviously, we could also simply fit our algorithm to the whole dataset, rather
than running cross-validation. This can be done by using the
:meth:`build_full_trainset()
<surprise.dataset.DatasetAutoFolds.build_full_trainset>` method which will
build a :class:`trainset <surprise.Trainset>` object:

.. literalinclude:: ../../examples/predict_ratings.py
    :caption: From file ``examples/predict_ratings.py``
    :name: predict_ratings.py
    :lines: 7-17

We can now predict ratings by directly calling the :meth:`predict()
<surprise.prediction_algorithms.algo_base.AlgoBase.predict>` method.  Let's say
you're interested in user 196 and item 302 (make sure they're in the
trainset!), and you know that the true rating :math:`r_{ui} = 4`:

.. literalinclude:: ../../examples/predict_ratings.py
    :caption: From file ``examples/predict_ratings.py``
    :name: predict_ratings2.py
    :lines: 20-24

The result should be:

.. parsed-literal::

    user: 196        item: 302        r_ui = 4.00   est = 4.06   {'actual_k': 40, 'was_impossible': False}

.. note::

    The :meth:`predict()
    <surprise.prediction_algorithms.algo_base.AlgoBase.predict>` uses **raw**
    ids (please read :ref:`this <raw_inner_note>` about raw and inner ids). As
    the dataset we have used has been read from a file, the raw ids are strings
    (even if they represent numbers).

We have so far used a built-in dataset, but you can of course use your own.
This is explained in the next section.

.. _load_custom:

Use a custom dataset
--------------------

`Surprise <https://nicolashug.github.io/Surprise/>`_ has a set of  builtin
:ref:`datasets <dataset>`, but you can of course use a custom dataset.
Loading a rating dataset can be done either from a file (e.g. a csv file), or
from a pandas dataframe.  Either way, you will need to define a :class:`Reader
<surprise.reader.Reader>` object for `Surprise
<https://nicolashug.github.io/Surprise/>`_ to be able to parse the file or the
dataframe.

.. _load_from_file_example:

- To load a dataset from a file (e.g. a csv file), you will need the
  :meth:`load_from_file() <surprise.dataset.Dataset.load_from_file>` method:

  .. literalinclude:: ../../examples/load_custom_dataset.py
      :caption: From file ``examples/load_custom_dataset.py``
      :name: load_custom_dataset.py
      :lines: 8-24

  For more details about readers and how to use them, see the :class:`Reader
  class <surprise.reader.Reader>` documentation.

  .. note::
      As you already know from the previous section, the Movielens-100k dataset
      is built-in so a much quicker way to load the dataset is to do ``data =
      Dataset.load_builtin('ml-100k')``. We will of course ignore this here.

.. _load_from_df_example:

- To load a dataset from a pandas dataframe, you will need the
  :meth:`load_from_df() <surprise.dataset.Dataset.load_from_df>` method. You
  will also need a :class:`Reader<surprise.reader.Reader>` object, but only
  the ``rating_scale`` parameter must be specified. The dataframe must have
  three columns, corresponding to the user (raw) ids, the item (raw) ids, and
  the ratings in this order. Each row thus corresponds to a given rating. This
  is not restrictive as you can reorder the columns of your dataframe easily.

  .. literalinclude:: ../../examples/load_from_dataframe.py
      :caption: From file ``examples/load_from_dataframe.py``
      :name: load_dom_dataframe.py
      :lines: 6-27

  The dataframe initially looks like this:

  .. parsed-literal::

            itemID  rating    userID
      0       1       3         9
      1       1       2        32
      2       1       4         2
      3       2       3        45
      4       2       1  user_foo


.. _use_cross_validation_iterators:

Use cross-validation iterators
------------------------------

For cross-validation, we can use the :func:`cross_validate()
<surprise.model_selection.validation.cross_validate>` function that does all
the hard work for us. But for a better control, we can also instantiate a
cross-validation iterator, and make predictions over each split using the
``split()`` method of the iterator, and the
:meth:`test()<surprise.prediction_algorithms.algo_base.AlgoBase.test>` method
of the algorithm. Here is an example where we use a classical K-fold
cross-validation procedure with 3 splits:

.. literalinclude:: ../../examples/use_cross_validation_iterators.py
    :caption: From file ``examples/use_cross_validation_iterators.py``
    :name: use_cross_validation_iterators.py
    :lines: 6-

Result could be, e.g.:

.. parsed-literal::
    RMSE: 0.9374
    RMSE: 0.9476
    RMSE: 0.9478

Other cross-validation iterator can be used, like LeaveOneOut or ShuffleSplit.
See all the available iterators :ref:`here <cross_validation_iterators_api>`.
The design of Surprise's cross-validation tools is heavily inspired from the
excellent scikit-learn API.

---------------------

.. _load_from_folds_example:

A special case of cross-validation is when the folds are already predefined by
some files. For instance, the movielens-100K dataset already provides 5 train
and test files (u1.base, u1.test ... u5.base, u5.test). Surprise can handle
this case by using a :class:`surprise.model_selection.split.PredefinedKFold`
object:

.. literalinclude:: ../../examples/load_custom_dataset_predefined_folds.py
    :caption: From file ``examples/load_custom_dataset_predefined_folds.py``
    :name: load_custom_dataset_predefined_folds.py
    :lines: 9-

Of course, nothing prevents you from only loading a single file for training
and a single file for testing. However, the ``folds_files`` parameter still
needs to be a ``list``.

.. _tuning_algorithm_parameters:

Tune algorithm parameters with GridSearchCV
-------------------------------------------

The :func:`cross_validate()
<surprise.model_selection.validation.cross_validate>` function reports accuracy
metric over a cross-validation procedure for a given set of parameters.  If you
want to know which parameter combination yields the best results, the
:class:`GridSearchCV <surprise.model_selection.search.GridSearchCV>` class
comes to the rescue.  Given a ``dict`` of parameters, this class exhaustively
tries all the combinations of parameters and reports the best parameters for any
accuracy measure (averaged over the different splits). It is heavily inspired
from scikit-learn's `GridSearchCV
<https://scikit-learn.org/stable/modules/generated/sklearn.model
_selection.GridSearchCV.html>`_.

Here is an example where we try different values for parameters ``n_epochs``,
``lr_all`` and ``reg_all`` of the :class:`SVD
<surprise.prediction_algorithms.matrix_factorization.SVD>` algorithm.

.. literalinclude:: ../../examples/grid_search_usage.py
    :caption: From file ``examples/grid_search_usage.py``
    :name: grid_search_usage.py
    :lines: 7-22

Result:

.. parsed-literal::

    0.961300130118
    {'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.4}

We are here evaluating the average RMSE and MAE over a 3-fold cross-validation
procedure, but any :ref:`cross-validation iterator
<cross_validation_iterators_api>` can used.

Once ``fit()`` has been called, the ``best_estimator`` attribute gives us an
algorithm instance with the optimal set of parameters, which can be used how we
please:

.. literalinclude:: ../../examples/grid_search_usage.py
    :caption: From file ``examples/grid_search_usage.py``
    :name: grid_search_usage2.py
    :lines: 24-26

.. _grid_search_note:
.. note::

    Dictionary parameters such as ``bsl_options`` and ``sim_options`` require
    particular treatment. See usage example below:

    .. parsed-literal::

        param_grid = {
            'k': [10, 20],
            'sim_options': {
                'name': ['msd', 'cosine'],
                'min_support': [1, 5],
                'user_based': [False],
            },
        }

    Naturally, both can be combined, for example for the
    :class:`KNNBaseline <surprise.prediction_algorithms.knns.KNNBaseline>`
    algorithm:

    .. parsed-literal::
        param_grid = {
            'bsl_options': {
                'method': ['als', 'sgd'],
                'reg': [1, 2],
            },
            'k': [2, 3],
            'sim_options': {
                'name': ['msd', 'cosine'],
                'min_support': [1, 5],
                'user_based': [False],
            },
        }

.. _cv_results_example:

For further analysis, the ``cv_results`` attribute has all the needed
information and can be imported in a pandas dataframe:

.. literalinclude:: ../../examples/grid_search_usage.py
    :caption: From file ``examples/grid_search_usage.py``
    :name: grid_search_usage3.py
    :lines: 30

In our example, the ``cv_results`` attribute looks like this (floats are
formatted):

.. parsed-literal::

    'split0_test_rmse': [1.0, 1.0, 0.97, 0.98, 0.98, 0.99, 0.96, 0.97]
    'split1_test_rmse': [1.0, 1.0, 0.97, 0.98, 0.98, 0.99, 0.96, 0.97]
    'split2_test_rmse': [1.0, 1.0, 0.97, 0.98, 0.98, 0.99, 0.96, 0.97]
    'mean_test_rmse':   [1.0, 1.0, 0.97, 0.98, 0.98, 0.99, 0.96, 0.97]
    'std_test_rmse':    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    'rank_test_rmse':   [7 8 3 5 4 6 1 2]
    'split0_test_mae':  [0.81, 0.82, 0.78, 0.79, 0.79, 0.8, 0.77, 0.79]
    'split1_test_mae':  [0.8, 0.81, 0.78, 0.79, 0.78, 0.79, 0.77, 0.78]
    'split2_test_mae':  [0.81, 0.81, 0.78, 0.79, 0.78, 0.8, 0.77, 0.78]
    'mean_test_mae':    [0.81, 0.81, 0.78, 0.79, 0.79, 0.8, 0.77, 0.78]
    'std_test_mae':     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    'rank_test_mae':    [7 8 2 5 4 6 1 3]
    'mean_fit_time':    [1.53, 1.52, 1.53, 1.53, 3.04, 3.05, 3.06, 3.02]
    'std_fit_time':     [0.03, 0.04, 0.0, 0.01, 0.04, 0.01, 0.06, 0.01]
    'mean_test_time':   [0.46, 0.45, 0.44, 0.44, 0.47, 0.49, 0.46, 0.34]
    'std_test_time':    [0.0, 0.01, 0.01, 0.0, 0.03, 0.06, 0.01, 0.08]
    'params':           [{'n_epochs': 5, 'lr_all': 0.002, 'reg_all': 0.4}, {'n_epochs': 5, 'lr_all': 0.002, 'reg_all': 0.6}, {'n_epochs': 5, 'lr_all': 0.005, 'reg_all': 0.4}, {'n_epochs': 5, 'lr_all': 0.005, 'reg_all': 0.6}, {'n_epochs': 10, 'lr_all': 0.002, 'reg_all': 0.4}, {'n_epochs': 10, 'lr_all': 0.002, 'reg_all': 0.6}, {'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.4}, {'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.6}]
    'param_n_epochs':   [5, 5, 5, 5, 10, 10, 10, 10]
    'param_lr_all':     [0.0, 0.0, 0.01, 0.01, 0.0, 0.0, 0.01, 0.01]
    'param_reg_all':    [0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6]

As you can see, each list has the same size of the number of parameter
combination. It corresponds to the following table:

==================  ==================  ==================  ================  ===============  ================  =================  =================  =================  ===============  ==============  ===============  ===============  ==============  ================  ===============  =================================================  ================  ==============  ===============
  split0_test_rmse    split1_test_rmse    split2_test_rmse    mean_test_rmse    std_test_rmse    rank_test_rmse    split0_test_mae    split1_test_mae    split2_test_mae    mean_test_mae    std_test_mae    rank_test_mae    mean_fit_time    std_fit_time    mean_test_time    std_test_time  params                                               param_n_epochs    param_lr_all    param_reg_all
==================  ==================  ==================  ================  ===============  ================  =================  =================  =================  ===============  ==============  ===============  ===============  ==============  ================  ===============  =================================================  ================  ==============  ===============
          0.99775             0.997744            0.996378          0.997291      0.000645508                 7           0.807862           0.804626           0.805282         0.805923      0.00139657                7          1.53341      0.0305216           0.455831      0.000922113  {'n_epochs': 5, 'lr_all': 0.002, 'reg_all': 0.4}                  5           0.002              0.4
          1.00381             1.00304             1.00257           1.00314       0.000508358                 8           0.816559           0.812905           0.813772         0.814412      0.00155866                8          1.5199       0.0367117           0.451068      0.00938646   {'n_epochs': 5, 'lr_all': 0.002, 'reg_all': 0.6}                  5           0.002              0.6
          0.973524            0.973595            0.972495          0.973205      0.000502609                 3           0.783361           0.780242           0.78067          0.781424      0.00138049                2          1.53449      0.00496203          0.441558      0.00529696   {'n_epochs': 5, 'lr_all': 0.005, 'reg_all': 0.4}                  5           0.005              0.4
          0.98229             0.982059            0.981486          0.981945      0.000338056                 5           0.794481           0.790781           0.79186          0.792374      0.00155377                5          1.52739      0.00859185          0.44463       0.000888907  {'n_epochs': 5, 'lr_all': 0.005, 'reg_all': 0.6}                  5           0.005              0.6
          0.978034            0.978407            0.976919          0.977787      0.000632049                 4           0.787643           0.784723           0.784957         0.785774      0.00132486                4          3.03572      0.0431101           0.466606      0.0254965    {'n_epochs': 10, 'lr_all': 0.002, 'reg_all': 0.4}                10           0.002              0.4
          0.986263            0.985817            0.985004          0.985695      0.000520899                 6           0.798218           0.794457           0.795373         0.796016      0.00160135                6          3.0544       0.00636185          0.488357      0.0576194    {'n_epochs': 10, 'lr_all': 0.002, 'reg_all': 0.6}                10           0.002              0.6
          0.963751            0.963463            0.962676          0.963297      0.000454661                 1           0.774036           0.770548           0.771588         0.772057      0.00146201                1          3.0636       0.0597982           0.456484      0.00510321   {'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.4}                10           0.005              0.4
          0.973605            0.972868            0.972765          0.973079      0.000374222                 2           0.78607            0.781918           0.783537         0.783842      0.00170855                3          3.01907      0.011834            0.338839      0.075346     {'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.6}                10           0.005              0.6
==================  ==================  ==================  ================  ===============  ================  =================  =================  =================  ===============  ==============  ===============  ===============  ==============  ================  ===============  =================================================  ================  ==============  ===============



Command line usage
------------------

Surprise can also be used from the command line, for example:

.. code::

    surprise -algo SVD -params "{'n_epochs': 5, 'verbose': True}" -load-builtin ml-100k -n-folds 3

See detailed usage by running:

.. code::

    surprise -h
