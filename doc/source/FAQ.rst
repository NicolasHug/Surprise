.. _FAQ:

FAQ
===

You will find here the Frequently Asked Questions, as well as some other
use-case examples that are not part of the User Guide.

How to get the top-N recommendations for each user
----------------------------------------------------------

Here is an example where we retrieve the top-10 items with highest
rating prediction for each user in the MovieLens-100k dataset. We first train
an SVD algorithm on the whole dataset, and then predict all the ratings for the
pairs (user, item) that are not in the training set. We then retrieve the
top-10 prediction for each user.

.. literalinclude:: ../../examples/top_n_recommendations.py
    :caption: From file ``examples/top_n_recommendations.py``
    :name: top_n_recommendations.py
    :lines: 8-

.. _precision_recall_at_k:

How to compute precision@k and recall@k
-----------------------------------------------------------------------

Here is an example where we compute Precision@k and Recall@k for each user:

:math:`\text{Precision@k} = \frac{ | \{ \text{Recommended items that are relevant} \} | }{ | \{ \text{Recommended items} \} | }`
:math:`\text{Recall@k} = \frac{ | \{ \text{Recommended items that are relevant} \} | }{ | \{ \text{Relevant items} \} | }`

An item is considered relevant if its true rating :math:`r_{ui}` is greater
than a given threshold.  An item is considered recommended if its estimated
rating :math:`\hat{r}_{ui}` is greater than the threshold, and if it is among
the k highest estimated ratings.

Note that in the edge cases where division by zero occurs, 
Precision@k and Recall@k values are undefined. 
As a convention, we set their values to 0 in such cases. 

.. literalinclude:: ../../examples/precision_recall_at_k.py
    :caption: From file ``examples/precision_recall_at_k.py``
    :name: precision_recall_at_k.py
    :lines: 5-

.. _get_k_nearest_neighbors:

How to get the k nearest neighbors of a user (or item)
--------------------------------------------------------------

You can use the :meth:`get_neighbors()
<surprise.prediction_algorithms.algo_base.AlgoBase.get_neighbors>` methods of
the algorithm object. This is only relevant for algorithms that use a
similarity measure, such as the :ref:`k-NN algorithms
<pred_package_knn_inpired>`.

Here is an example where we retrieve the 10 nearest neighbors of the movie Toy
Story from the MovieLens-100k dataset. The output is:

.. parsed-literal::

    The 10 nearest neighbors of Toy Story are:
    Beauty and the Beast (1991)
    Raiders of the Lost Ark (1981)
    That Thing You Do! (1996)
    Lion King, The (1994)
    Craft, The (1996)
    Liar Liar (1997)
    Aladdin (1992)
    Cool Hand Luke (1967)
    Winnie the Pooh and the Blustery Day (1968)
    Indiana Jones and the Last Crusade (1989)

There's a lot of boilerplate because of the conversions between movie names and
their raw/inner ids (see :ref:`this note <raw_inner_note>`), but it all boils
down to the use of :meth:`get_neighbors()
<surprise.prediction_algorithms.algo_base.AlgoBase.get_neighbors>`:

.. literalinclude:: ../../examples/k_nearest_neighbors.py
    :caption: From file ``examples/k_nearest_neighbors.py``
    :name: k_nearest_neighbors.py
    :lines: 9-

Naturally, the same can be done for users with minor modifications.

.. _serialize_an_algorithm:

How to serialize an algorithm
-----------------------------

Prediction algorithms can be serialized and loaded back using the :func:`dump()
<surprise.dump.dump>` and :func:`load() <surprise.dump.load>` functions. Here
is a small example where the SVD algorithm is trained on a dataset and
serialized. It is then reloaded and can be used again for making predictions:

.. literalinclude:: ../../examples/serialize_algorithm.py
    :caption: From file ``examples/serialize_algorithm.py``
    :name: serialize_algorithm.py
    :lines: 7-

.. _further_analysis:

Algorithms can be serialized along with their predictions, so that can be
further analyzed or compared with other algorithms, using pandas dataframes.
Some examples are given in the two following notebooks:

    * `Dumping and analysis of the KNNBasic algorithm
      <https://nbviewer.jupyter.org/github/NicolasHug/Surprise/tree/master/examples/notebooks/KNNBasic_analysis.ipynb/>`_.
    * `Comparison of two algorithms
      <https://nbviewer.jupyter.org/github/NicolasHug/Surprise/tree/master/examples/notebooks/Compare.ipynb/>`_.

How to build my own prediction algorithm
----------------------------------------

There's a whole guide :ref:`here<building_custom_algo>`.

.. _raw_inner_note:

What are raw and inner ids
--------------------------

Users and items have a raw id and an inner id. Some methods will use/return a
raw id (e.g. the :meth:`predict()
<surprise.prediction_algorithms.algo_base.AlgoBase.predict>` method), while
some other will use/return an inner id.

Raw ids are ids as defined in a rating file or in a pandas dataframe. They can
be strings or numbers. Note though that if the ratings were read from a file
which is the standard scenario, they are represented as strings. **This is
important to know if you're using e.g.** :meth:`predict()
<surprise.prediction_algorithms.algo_base.AlgoBase.predict>` **or other methods
that accept raw ids as parameters.**

On trainset creation, each raw id is mapped to a unique integer called inner
id, which is a lot more suitable for `Surprise
<https://nicolashug.github.io/Surprise/>`_ to manipulate. Conversions between
raw and inner ids can be done using the :meth:`to_inner_uid()
<surprise.Trainset.to_inner_uid>`, :meth:`to_inner_iid()
<surprise.Trainset.to_inner_iid>`, :meth:`to_raw_uid()
<surprise.Trainset.to_raw_uid>`, and :meth:`to_raw_iid()
<surprise.Trainset.to_raw_iid>` methods of the :class:`trainset
<surprise.Trainset>`.


Can I use my own dataset with Surprise, and can it be a pandas dataframe
------------------------------------------------------------------------

Yes, and yes. See the :ref:`user guide <load_custom>`.

How to tune an algorithm parameters
-----------------------------------

You can tune the parameters of an algorithm with the :class:`GridSearchCV
<surprise.model_selection.search.GridSearchCV>` class as described :ref:`here
<tuning_algorithm_parameters>`. After the tuning, you may want to have an
:ref:`unbiased estimate of your algorithm performances
<unbiased_estimate_after_tuning>`.

How to get accuracy measures on the training set
------------------------------------------------

You can use the :meth:`build_testset()
<surprise.Trainset.build_testset()>` method of the :class:`Trainset
<surprise.Trainset>` object to build a testset that can be then used
with the :meth:`test()
<surprise.prediction_algorithms.algo_base.AlgoBase.test>` method:

.. literalinclude:: ../../examples/evaluate_on_trainset.py
    :caption: From file ``examples/evaluate_on_trainset.py``
    :name: evaluate_on_trainset.py
    :lines: 7-21

Check out the example file for more usage examples.

.. _unbiased_estimate_after_tuning:

How to save some data for unbiased accuracy estimation
------------------------------------------------------

If your goal is to tune the parameters of an algorithm, you may want to spare a
bit of data to have an unbiased estimation of its performances. For instance
you may want to split your data into two sets A and B. A is used for parameter
tuning using grid search, and B is used for unbiased estimation. This can be
done as follows:

.. literalinclude:: ../../examples/split_data_for_unbiased_estimation.py
    :caption: From file ``examples/split_data_for_unbiased_estimation.py``
    :name: split_data_for_unbiased_estimation.py
    :lines: 8-

How to have reproducible experiments
------------------------------------

Some algorithms randomly initialize their parameters (sometimes with
``numpy``), and the cross-validation folds are also randomly generated. If you
need to reproduce your experiments multiple times, you just have to set the
seed of the RNG at the beginning of your program:

.. code::

    import random
    import numpy as np

    my_seed = 0
    random.seed(my_seed)
    np.random.seed(my_seed)

.. _data_folder:

Where are datasets stored and how to change it?
-----------------------------------------------

By default, datasets downloaded by Surprise will be saved in the
``'~/.surprise_data'`` directory. This is also where dump files will be stored.
You can change the default directory by setting the ``'SURPRISE_DATA_FOLDER'``
environment variable.

Can Surprise support content-based data or implicit ratings?
------------------------------------------------------------

No: this is out of scope for surprise. Surprise was designed for explicit
ratings.
