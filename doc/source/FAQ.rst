.. _FAQ:

FAQ
===

.. _get_k_nearest_neighbors:

How to get the :math:`k` nearest neighbors of a user (or item)
--------------------------------------------------------------

You can use the :meth:`get_neighbors()
<surprise.prediction_algorithms.algo_base.AlgoBase.get_neighbors>` methods of
the algorithm. This is only relevent for algorithms using a similarity measure,
such as the :ref:`k-NN algorithms <pred_package_knn_inpired>`.

Here is an example where we retrieve the k-nearest neighbors of the movie Toy
Story from the MovieLens-100k dataset. The same can be done for users with
minor changes.  There's a lot of boilerplate because of the id conversions, but
it all boils down to the use of ``get_neighbors()``:

.. literalinclude:: ../../examples/k_nearest_neighbors.py
    :caption: From file ``examples/k_nearest_neighbors.py``
    :name: k_nearest_neighbors.py
    :lines: 10-

How to get the top-:math:`k` recommendations for a user
-------------------------------------------------------

.. _save_algorithm_for_later_use:

How to serialize an algorithm
-----------------------------

Prediction algortihms can be serialized and loaded back using the :func:`dump()
<surprise.dump.dump>` and :func:`load() <surprise.dump.load>` functions. Here
is a small example where the SVD algorithm is trained on a dataset and
serialized. It is then reloaded and can be used again for making predictions:

.. literalinclude:: ../../examples/serialize_algorithm.py
    :caption: From file ``examples/serialize_algorithm.py``
    :name: serialize_algorithm.py
    :lines: 9-

How to build my own prediction algorithm
----------------------------------------

See the :ref:`user guide <building_custom_algo>`.

What are raw and inner ids
--------------------------

See :ref:`this note <raw_inner_note>`.

How to use my own dataset with Surprise
---------------------------------------

See the :ref:`user guide <load_custom>`.

How to tune an algorithm parameters
-----------------------------------

To tune the parameters of your algorithm, you can use the :class:`GridSearch
<surprise.evaluate.GridSearch>` class as described :ref:`here
<tuning_algorithm_parameters>`. After the tuning, you may want to have an
:ref:`unbiased estimate of your algorithm performances
<unbiased_estimate_after_tuning>`.

How to get accuracy measures on the training set
------------------------------------------------

You can use the :meth:`build_testset()
<surprise.dataset.Trainset.build_testset()>` method of the :class:`Trainset
<surprise.dataset.Trainset>` object to build a testset that can be then used
with the :meth:`test()
<surprise.prediction_algorithms.algo_base.AlgoBase.test>` method:

.. literalinclude:: ../../examples/evaluate_on_trainset.py
    :caption: From file ``examples/evaluate_on_trainset.py``
    :name: evaluate_on_trainset.py
    :lines: 9-24

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
    :lines: 10-
