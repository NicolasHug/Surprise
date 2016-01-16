TODO
====

Court terme :
-------------

* set up a nice API (in progress)
* handle big datasets (memory error when computing similarities)
* write some tests (in progress)
* handle algo-specific or similarity-specific parameters (such as 'k' for knn,
  regularization parameters, shrinkage paramaters, etc.) in an appropriate
  manner, rather than pass them all to constructors... UPDATE: ok so using
  kwargs like matplotlib.pyplot might be enough. should we create a
  'Similarity' class?

Long terme :
------------

* handle more datasets + user-defined datasets
* implement matrix factorization algorithms (PF)
* document every usefull piece of code
* set up algorithm comparation tools (first draft done)
* package everything properly for future open sourcing

Done:
-----

* clean the main and all the dataset handling stuff (still needs to be
  polished)
* rewrite this TODO in english
* create a proper project structure
* from camelCase to snake\_case
