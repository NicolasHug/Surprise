TODO
====

* keep on testing
* keep on documenting and commenting code
* Matrix facto algo
* create option in __main__ to clean the .pyrec directory

Some random observations
------------------------

* maybe clean a little all the dataset machinery? Plus, are the
    raw2inner_id_users and raw2inner_id_items worth keeping? May be for
    analysing tools, I don't know right now.

Done:
-----

* set up a nice API (looks ok now)
* handle algo-specific or similarity-specific parameters (such as 'k' for knn,
  regularization parameters, shrinkage paramaters, etc.) in an appropriate
  manner, rather than pass them all to constructors... UPDATE: ok so using
  kwargs like matplotlib.pyplot might be enough. should we create a
  'Similarity' class?
* clean the main and all the dataset handling stuff (still needs to be
  polished)
* rewrite this TODO in english
* create a proper project structure
* from camelCase to snake\_case
