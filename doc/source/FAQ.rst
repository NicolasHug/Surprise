.. _FAQ:

FAQ
===

I want to get accuracy measures on the training set
---------------------------------------------------

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
