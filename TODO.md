TODO
====

* create option in __main__ to clean the .recsys directory. Actually, the
  __main__ module should be entirely reviewed.
* Add a 'min_support' parameter to sim_options?
* should a Prediction output the raw id or the inner id? Right now it's the
  inner id. Maybe sort this out when working on the comparison tools.
* when dumping, we should dump all the algorithm parameter. Use __dict__ ?
* test all algorithms with a user with no ratings and an items with no ratings.
  (linked to a todo in the predict method)


Done:
-----

* remove kwargs : done where useless.
* say something quick about baseline computation (when not matrix facto) 
* Matrix facto algo
* allow the 'estimate' method to return some details about prediction (such as
  the number of neighbors for a KNN)
* allow to train on a SINGLE file without test set, and let user query for some
  predictions
* write tuto for using only predict() (and not test)
* maybe clean a little all the dataset machinery? Plus, are the
  raw2inner_id_users and raw2inner_id_items worth keeping? May be for analysing
  tools, I don't know right now. EDIT: yes, we need to keep them, simply
  because the similarity computation can only work with integer as indexes
  (numpy arrays).
* sort out this warning issue coming from cython
* say something about the sim > 0 in knns algos
* get less restrictive requirements.txt
* write the custom algorithm tutorial
* improve test coverage
* add the cool stickers on the readme just like scikit learn
* set up travis
* keep on testing
* keep on documenting and commenting code
* extensively test the reader class, + check that the doc is OK for reader
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
