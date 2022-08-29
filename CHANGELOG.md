VERSION 1.1.1
=============

Date: 19/07/2020

Mostly doc typos and some minor bug fixes. Future versions (if any)
will not support Python 2 anymore.

Bug Fixes
---------

* 'mse' is now available in GridSearCV and RandomizedSearchCV
* The Jester dataset link was updated
* Fixed a potential race condition when creating dataset directories


VERSION 1.1.0
=============

Date: 13/11/2019

1.1.0 will be the last stable version with new features. Next versions will
only provide bug-fixes, but no new features. (And probably not support
Python 2 at all).

Enhancements
------------

* The prompt confirmation can now be disabled when downloading a dataset.
* The MSE metric has been added.

Bug Fixes
---------

* Fixed a bug where msd and peasron would not properly set the similarity to
  zero when ``min_support`` wasn't reached.

API Changes
-----------

* Tools that were deprecated before (data.split(), GridSearch, evaluate) are
  now removed.

VERSION 1.0.6
=============

Date: 22/04/18

Enhancements
------------

* Added verbose option to algorithms using a similarity matrix or baseline
  computation, to avoid unwanted printed messages.
* When PredictionImpossible is raised, the prediction is now deferred to
  default_prediction() method, which can be overridden is child classes. This
  allows to not always set the default prediction to the average rating, which
  can be useful for some algorithms (e.g. those working with implicit positive
  feedback).
* LeaveOneOut() now accepts a min_n_ratings parameter to make sure users in the
  trainset have at least min_n_ratings ratings.
* Dumping is now done with pickle's highest protocol which allows for larger
  files.

Bug Fixes
---------

* Joblib parameter `n_jobs` now defaults to 1 (no use of multiprocessing).
  Should fix issues with Windows users.
* `cross_validate` now returns correct values for training measures (used to
  return test measures instead).

VERSION 1.0.5
=============

Date: 09/01/18

Enhancements
------------

* Cross-validation tools have been entirely reworked. We can now rely on
  powerful and flexible cross-validation iterators, inspired by scikit-learn's
  API.
* the evaluate() method has been replaced by cross-validate which is parallel
  and can return measures on trainset as well as computation times.
* GridSearch is now parallel, using joblib.
* GridSearch now allows to refit an algorithm on the whole dataset.
* default data directory can now be custom with env variable
  SURPRISE_DATA_FOLDER
* the fit() (and train()) methods now return self, which allows one-liners like
  algo.fit(trainset).test(testset)
* Algorithms using a random initialization (e.g. SVD, NMF, CoClustering) now
  have a random_state parameter for seeding the RNG.
* The getting started guide has been rewritten

API Changes
-----------

* The train() method is now deprecated and replaced by the fit() method (same
  signature). Calls to train() should still work as before.
* Using data.split() or accessing the data.folds() generator is deprecated and
  replaced by the use of the more powefull CV iterators.
* evaluate() is deprecated and  replaced by model_selection.cross_validate(),
  which is parallel.
* GridSearch is deprecated and replaced by model_selection.GridSearchCV()

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
