TODO
====

current issues:
* binarize should allow a 'keep_negative' parameter. Else stuff like
  LeaveOneOut would not work as expected.
* rating_scale param of trainset makes no sense when using a dataset that has
  been binarized. E.g. it will still be (1, 5) for a binarized movielens. Ugly
  solution: change self.reader.rating_scale in Dataset.binarize. Should find
  out a better way to do this. Maybe get rid of rating_scale entirely?
* It makes no sense to clip ratings in predict() (plus, as the rating_scale
  parameter is used and is not correct, it makes even less sense) for algorithm
  reurning a score and not a rating. Solution: test() method should allow to
  pass parameters to predict() as well as cross_validate() and GridSearch(). I
  REMOVED CLIPPING BY DEFAULT TO TEST BPR BUT IT SHOULD BE PUT BACK.
* How can AUC measure fit within current tools (cross_validate, etc.)? Can it
  be used with other CV iterators than just LeaveOneOut?
* Using anti_testset() for AUC computation is not correct. anti_testset() will
  include the item that was in the testset (positive feedback), and we are only
  interested in items with negative feedback.


* convert lists in ur and ir into sets for quicker look-up?
* put references in the algorithm page rather than in the ref page.
* in Trainset, create ur by default and create ir only if needed?
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
