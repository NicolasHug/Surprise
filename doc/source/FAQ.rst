.. _FAQ:

FAQ
===

How to get the :math:`k` nearest neighbors of a user (or item)
--------------------------------------------------------------

How to get the top-:math:`k` recommendations for a user
-------------------------------------------------------

How to save an algorithm for later use
--------------------------------------

How to build my own prediction algorithm
----------------------------------------

What are raw and inner ids
--------------------------

How to use my own dataset with Surprise
---------------------------------------

How to tune an algorithm parameters
-----------------------------------

How to get accuracy measures on the training set
------------------------------------------------

You can use the :meth:`build_testset()
<surprise.dataset.Trainset.build_testset()>` method of the :class:`Trainset
<surprise.dataset.Trainset>` object to build a trainset that can be then used
with the :meth:`test()
<surprise.prediction_algorithms.algo_base.AlgoBase.test>` method:

.. literalinclude:: ../../examples/evaluate_on_trainset.py
    :caption: From file ``examples/evaluate_on_trainset.py``
    :name: evaluate_on_trainset.py
    :lines: 9-24

Check out the example file for more usage examples.

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
