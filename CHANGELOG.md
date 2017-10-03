VERSION 1.0.5 (latest, in development)
======================================

Enhancements
------------

* GridSearch is now parallel, using joblib.

VERSION 1.0.4
=============

Date: 20/09/17

Enhancements
------------

* Added possibility to load a dataset from a pandas dataframe
* Added Precision and Recall examples to the FAQ (Maher Malaeb)
* Added a kNN algorithm with normalization by z-score (Hengji Liu)
* kNN algorithms now use heapq instead of list.sort() (computation time
  enhancement for large datasets).

Fixes
-----

* Prediciont.__str__() when r_ui is None
* GridSearch for dict parameters is now working as expected

API Changes
-----------

* param_grid for GridSearch is now slightly different for dict parameters (see
  note on [the
  docs](http://surprise.readthedocs.io/en/stable/getting_started.html#tune-algorithm-parameters-with-gridsearch)).

VERSION 1.0.3
=============

Date: 03/05/17

Enhancements
------------

* Added FAQ in the doc
* Added the possibility to retrieve the k nearest neighbors of a user or an
  item.
* Changed the dumping process a bit (see API changes). Plus, dumps can now be
  loaded.
* Added possibility to build a testset from the ratings of a training set
* Added inner-to-raw id conversion in the Trainset class
* The r_ui parameter of the predict() method is now optional

Fixes
-----
* Fixed verbosity of the evaluate function
* Corrected prediction when only user (or only item) is unknown in SVD and NMF
  algorithms. Thanks to kenoung!
* Corrected factor vectors initialization of SVD algorithms. Thanks to
  adideshp!

API Changes
-----------

* The dump() method now dumps a list of predition (optional) and an algorithm
  (optional as well). The algorithm is now a real algorithm object. The
  trainset is not dumped anymore as it is already part of the algorithm anyway.
* The dump() method is now part of the dump namespace, and not the global
  namespace (so it is accessed by surprise.dump.dump)

VERSION 1.0.2
=============

Date: 04/01/17

Just a minor change so that README.md is converted to rst for better rendering
on PyPI.

VERSION 1.0.1
=============

Date: 02/01/17

Enhancements
------------

* Added the GridSearch feature, by Maher
* Added a 'clip' option to the predict() method
* Added NMF algorithm
* Added entry point for better command line usage.
* Added CoClustering algorithm.
* Added SlopeOne algorithm.
* Added Probabilistic Matrix Factorization as an option SVD
* Cythonized Baseline Computation

Other
-----

* Surprise is now a scikit!
* Changed license to BSD
* Six is now a dependency

VERSION 1.0.0
=============

Date: 22/11/16

* Changed name from recsys to surprise
* Improved printing of accuracy measures.
* Added version number.
* Rewrote the the __main__.py

VERSION 0.0.4
=============

Date: 15/11/16

Enhancements
------------

* Added notebooks for comparing and evaluating algorithm performances
* Better use of setup.py
* Added a min_support parameter to the similarity measures.
* Added a min_k parameter to the KNN algorithms.
* The similarity matrix and baselines are now returned.
* You can now train on a whole training set without test set.
* The estimate method can return a tuple with prediction details.
* Added SVD and SVD++ algorithms.
* Removed all the x/y vs user/item stuff. That was useless for most algorithms.


API Changes
-----------

* Removed the @property decorator for many iterators.
* It's now up to the algorithms to decide if they can or cannot make a
	prediction.

VERSION 0.0.3
=============

Date: 25/10/16

* Added support for Python 2
