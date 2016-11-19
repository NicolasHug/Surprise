.. RecSys documentation master file, created by
   sphinx-quickstart on Tue Dec 29 20:08:18 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _index:

Welcome to RecSys' documentation!
=================================

`RecSys <https://niourf.github.io/RecSys/>`_ is an open source Python library 
that provides with tools to build and evaluate the performance of many
recommender system prediction algorithms. Its goal is to make life easy(-ier)
for reseachers, teachers and students who want to play around with new
recommender algorithms ideas and teach/learn more about recommender systems.

`RecSys <https://niourf.github.io/RecSys/>`_ was designed with the following
purposes in mind:

- Give the user perfect control over his experiments. To this end, a strong
  emphasis is laid on :ref:`documentation <index>`, which we
  have tried to make as clear and precise as possible by pointing out every
  details of the algorithms.
- Alleviate the pain of :ref:`dataset handling <load_custom>`. Users can use
  both *built-in* datasets
  (`Movielens <http://grouplens.org/datasets/movielens/>`_,
  `Jester <http://eigentaste.berkeley.edu/dataset/>`_), and their own *custom*
  datasets.
- Provide with various ready-to-use :ref:`prediction
  algorithms <prediction_algorithms_package>`.
- Make it easy to implement :ref:`new algorithm
  ideas <building_custom_algo>`.
- Provide with tools to :func:`evaluate <recsys.evaluate.evaluate>`, `analyse
  <http://nbviewer.jupyter.org/github/Niourf/RecSys/tree/master/examples/notebooks/KNNBasic_analysis.ipynb/>`_
  and `compare
  <http://nbviewer.jupyter.org/github/Niourf/RecSys/tree/master/examples/notebooks/Compare.ipynb/>`_
  the algorithms performance. Cross-validation procedures can be run very
  easily.

.. toctree::
   :caption: User Guide

   getting_started
   notation_standards
   prediction_algorithms
   building_custom_algo


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   prediction_algorithms_package
   similarities
   accuracy
   dataset
   evaluate
   dump


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
