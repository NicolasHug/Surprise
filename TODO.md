TODO
====

* remove offset from everywhere (Reader, Trainset, test(), predict()...). Make
  a test to make sure that zero ratings are handled correctly. Right now we
  store ratings as defaultdict(list) in ur and ir, so there's no problem. Maybe
  it will change in the future, we would have to find a workaround.
* rating_scale should be specified on dataset creation: load_from_file,
  load_from_folds, load_from_df. Deprecate its use from rating but fall back to
  it if it has only been specified here.
* Dataset.binarize should change rating_scale to something else. Maybe 'unary'?
* about clip: don't clip if rating_scale is 'unary'?
* test(), cross_validate() and GridSearch.fit() should probably allow a
  'predict_params' parameter?
* What about a 'dont_clip' attribute in algorithms that could take precedence
  over the clipping behaviour? What about just overriding predict() in the
  child classes?


* Document get_dataset_dir()...
* make some filtering dataset tools, like remove users/items with less/more
  than n ratings, binarize a dataset, etc...
* then implement MFBPR and see how it goes
* Allow incremental updates for some algorithms

Done:
-----

* Grid search now has the refit param.
* Grid search and cross_validate now allow return_train_score
* Make all fit methods return self. Update docs on building custom algorithms
* Update doc of MF algo to indicate how to retrieve latent factors.
* all algorithms using random initialization now have a random_state parameter.
* CV iterators:
  - Write basic CV iterators
  - evaluate -> rewrite to use CV iterators. Rename it into cross_validate.
  - Same for GridSearch. Keep it in a model_selection module like scikit-learn
    so that we can keep the old deprecated version. 
  - Make cross validation parallel with joblib
  - Add deprecation warnings for evaluate and GridSearch()
  - handle the cv_results attribute for grid search
  - (re)write all verbose settings for gridsearch and cross_validate
  - Change examples so they use CV iterators and the new gridsearch and
    cross_validate
  - indicate in docs that split(), folds(), evaluate() and gridsearch() are
    deprecated
  - Write comments, docstring and update all docs
  - Update main and command-line usage doc in getting started.rst
* Allow to change data folder from env variable
* Complete FAQ
* Change the dumping machinery to be more consistent 
* Allow to test on the trainset
* make bibtex entry
* Verbosity of gridsearch still prints stuff because of evaluate. Fix that.
* Make the r_ui param of predict optional
* Put some PredictionImpossible messages in every algo
* allow a 'clip' option to the predict method? Also, describe r_min and r_max
* configure entrypoints to use surprise directly from command line
* Allow a 'biased' option in the SVD algo. If true, use baselines, if False,
  don't. It should be pretty easy to do.
* create option in __main__ to clean the .recsys directory. Actually, the
  __main__ module should be entirely reviewed.
* when dumping, we should dump all the algorithm parameter. Use __dict__ ?
* do something about the generators Python 2 vs 3 (range, dict.items(), etc...)
* should a Prediction output the raw id or the inner id? Right now it's the
  inner id. Maybe sort this out when working on the comparison tools.
* allow the perf dict returned by evaluate to accept keys with lower/upper
  case for retarded users such as me.
* Add a 'min_support' parameter to sim_options? Add a min_k to knns?
* Do something about the user_based stuff. It should be better. Check knns BTW.
* Do something about unknown users and unknown items, i.e. users or items that
  have no rating in the trainset. Right now, the predict method checks if the
  name starts with 'unknown' but this is shiiite because it's dependent on the
  construct_trainset method, which is sometimes never called (so the raw2inner
  stuff will come in play somehow). Plus, It should be up to the algorithms to
  choose whether it can (or can't) make a prediction even if user or item is
  unknown.
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
