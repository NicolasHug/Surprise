Current
=======

Enhancements
------------
* Added CoClustering algorithm.
* Added SlopeOne algorithm.
* Added Probabilistic Matrix Factorization as an option SVD
* Cythonized Baseline Computation

Other
-----
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

* Removed the @property decorator of for many iterators.
* It's now up to the algorithms to decide if they can or cannot make a
	prediction.

VERSION 0.0.3
=============

Date: 25/10/16

* Added support for Python 2
