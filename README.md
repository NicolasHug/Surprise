[![GitHub version](https://badge.fury.io/gh/nicolashug%2FSurprise.svg)](https://badge.fury.io/gh/nicolashug%2FSurprise)
[![Documentation Status](https://readthedocs.org/projects/surprise/badge/?version=latest)](http://surprise.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/NicolasHug/Surprise.svg?branch=master)](https://travis-ci.org/NicolasHug/Surprise)
[![python versions](https://img.shields.io/badge/python-2.7%2C%203.5-blue.svg)](http://surpriselib.com)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)




Surprise
========

Overview
--------

[Surprise](http://surpriselib.com) is an easy-to-use open
source Python library for recommender systems. Its goal is to make life easier
for reseachers who want to play around with new algorithms ideas, for teachers
who want some teaching materials, and for students.

[Surprise](http://surpriselib.com) **was designed with the
following purposes in mind**:

- Give the user perfect control over his experiments. To this end, a strong
  emphasis is laid on
  [documentation](http://surprise.readthedocs.io/en/latest/index.html), which we
  have tried to make as clear and precise as possible by pointing out every
  details of the algorithms.
- Alleviate the pain of [Dataset
  handling](http://surprise.readthedocs.io/en/latest/getting_started.html#load-a-custom-dataset).
  Users can use both *built-in* datasets
  ([Movielens](http://grouplens.org/datasets/movielens/),
  [Jester](http://eigentaste.berkeley.edu/dataset/)), and their own *custom* datasets.
- Provide various ready-to-use [prediction
  algorithms](http://surprise.readthedocs.io/en/latest/prediction_algorithms_package.html)
  (see below) [similarity
  measures](http://surprise.readthedocs.io/en/latest/similarities.html)
  (cosine, MSD, pearson...).
- Make it easy to implement [new algorithm
  ideas](http://surprise.readthedocs.io/en/latest/building_custom_algo.html).
- Provide tools to [evaluate](http://surprise.readthedocs.io/en/latest/evaluate.html),
  [analyse](http://nbviewer.jupyter.org/github/NicolasHug/Surprise/tree/master/examples/notebooks/KNNBasic_analysis.ipynb/)
  and
  [compare](http://nbviewer.jupyter.org/github/NicolasHug/Surprise/blob/master/examples/notebooks/Compare.ipynb)
  the algorithms performance. Cross-validation procedures can be run very easily.

At the moment, the available prediction algorithms are:

- [NormalPredictor](http://surprise.readthedocs.io/en/latest/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor):
  an algorithm predicting a random rating based on the distribution of the
  training set, which is assumed to be normal.
- [BaselineOnly](http://surprise.readthedocs.io/en/latest/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly):
  an agorithm predicting the baseline estimate for given user and item.
- [KNNBasic](http://surprise.readthedocs.io/en/latest/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic):
  a basic collaborative filtering algorithm.
- [KNNWithMeans](http://surprise.readthedocs.io/en/latest/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans):
  a basic collaborative filtering algorithm, taking into account the mean
  ratings of each user.
- [KNNBaseline](http://surprise.readthedocs.io/en/latest/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline):
  a basic collaborative filtering algorithm taking into account a baseline
  rating.
- [SVD](http://surprise.readthedocs.io/en/latest/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)
  and
  [PMF](http://surprise.readthedocs.io/en/latest/matrix_factorization.html#unbiased-note):
  the famous SVD algorithm, as popularized by Simon Funk during the Netflix
  Prize. The unbiased version is equivalent to Probabilistic Matrix
  Factorization.
- [SVD++](http://surprise.readthedocs.io/en/latest/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp):
  an extension of SVD taking into account implicite ratings.
- [NMF](http://surprise.readthedocs.io/en/latest/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF):
  a collaborative filtering algorithm based on Non-negative Matrix
  Factorization. (Available in latest version).
- [Slope One](http://surprise.readthedocs.io/en/latest/slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne): a simple yet accurate collaborative filtering algorithm. (Available in latest version).
- [Co-clustering](http://surprise.readthedocs.io/en/latest/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering): a collaborative filtering algorithm based on co-clustering. (Available in latest version).


The name *SurPRISE* (roughly :) ) stands for Simple Python RecommendatIon
System Engine.

Installation / Usage
--------------------

The easiest way is to use pip (you'll need [numpy](http://www.numpy.org/)):

    $ pip install surprise

Or you can clone the repo and build the source (you'll need
[Cython](http://cython.org/) and [numpy](http://www.numpy.org/)):

    $ git clone https://github.com/NicolasHug/surprise.git
    $ python setup.py install

Example
-------

Here is a simple example showing how you can (down)load a dataset, split it for
3-folds cross-validation, and compute the MAE and RMSE of the
[SVD](http://surprise.readthedocs.io/en/latest/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)
algorithm.

```python
from surprise import SVD
from surprise import Dataset
from surprise import evaluate


# Load the movielens-100k dataset (download it if needed),
# and split it into 3 folds for cross-validation.
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print(perf)
```

**Output**:

```
Evaluating RMSE, MAE of algorithm SVD.

        Fold 1  Fold 2  Fold 3  Mean
MAE     0.7475  0.7447  0.7425  0.7449
RMSE    0.9461  0.9436  0.9425  0.9441
```

Surprise can also be used from the command line, e.g.:

```
python -m surprise -algo SVD -params "{'n_factors': 10}"  -load-builtin ml-100k -n-folds 3
```

Benchmarks
----------

Here are the average RMSE, MAE and total execution time of various algorithms
(with their default parameters) on a 5-folds cross-validation procedure. The
datasets are the [Movielens](http://grouplens.org/datasets/movielens/) 100k and
1M datasets. The folds are the same for all the algorithms (the random seed is
set to 0). All experiments are run on a small laptop with Intel Core i3 1.7
GHz, 4Go RAM. The execution time is the *real* execution time, as returned by
the GNU [time](http://man7.org/linux/man-pages/man1/time.1.html) command.

|  [Movielens 100k](http://grouplens.org/datasets/movielens/100k) |  RMSE  |   MAE  | Time (s) |
|-----------------|:------:|:------:|:--------:|
| [NormalPredictor](http://surprise.readthedocs.io/en/latest/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor) | 1.5228 | 1.2242 |     4    |
| [BaselineOnly](http://surprise.readthedocs.io/en/latest/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly)    |  .9445 |  .7488 |    5    |
| [KNNBasic](http://surprise.readthedocs.io/en/latest/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic)        |  .9789 |  .7732 |    27    |
| [KNNWithMeans](http://surprise.readthedocs.io/en/latest/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans)    |  .9514 |  .7500 |    30    |
| [KNNBaseline](http://surprise.readthedocs.io/en/latest/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline)     |  .9306 |  .7334 |    44    |
| [SVD](http://surprise.readthedocs.io/en/latest/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)             |  .9392 |  .7409 |    46    |
| [SVD++](http://surprise.readthedocs.io/en/latest/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp)             |  .9200 |  .7253 |    31min    |
| [NMF](http://surprise.readthedocs.io/en/latest/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF)             |  .9634 |  .7572 |    55    |
| [Slope One](http://surprise.readthedocs.io/en/latest/slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne)             |  .9454 |  .7430 |    25    |
| [Co clustering](http://surprise.readthedocs.io/en/latest/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering)             |  .9678 |  .7579 |    15    |


|  [Movielens 1M](http://grouplens.org/datasets/movielens/1m) |  RMSE  |   MAE  | Time (min) |
|-----------------|:------:|:------:|:--------:|
| [NormalPredictor](http://surprise.readthedocs.io/en/latest/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor) | 1.5037 | 1.2051 |     < 1    |
| [BaselineOnly](http://surprise.readthedocs.io/en/latest/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly)    |  .9086 | .7194 |    < 1    |
| [KNNBasic](http://surprise.readthedocs.io/en/latest/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic)        |  .9207 |  .7250 |    22    |
| [KNNWithMeans](http://surprise.readthedocs.io/en/latest/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans)    |  .9292 |  .7386 |    22    |
| [KNNBaseline](http://surprise.readthedocs.io/en/latest/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline)     |  .8949 | .7063 |    44    |
| [SVD](http://surprise.readthedocs.io/en/latest/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)             |  .8936 |  .7057 |    7    |
| [NMF](http://surprise.readthedocs.io/en/latest/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF)             |  .9155 |  .7232 |    9    |
| [Slope One](http://surprise.readthedocs.io/en/latest/slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne)             |  .9065 |  .7144 |    8    |
| [Co clustering](http://surprise.readthedocs.io/en/latest/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering)             |  .9155 |  .7174 |    2    |


Documentation, Getting Started
------------------------------

The documentation with many other usage examples is [available
online](http://surprise.readthedocs.io/en/latest/index.html) on ReadTheDocs.

License
-------

This project is licensed under the [BSD
3-Clause](https://opensource.org/licenses/BSD-3-Clause) license.

Acknowledgements:
----------------

- [Pierre-François Gimenez](https://github.com/PFgimenez), for his valuable
  insights on software design.

Contributing, feedback
----------------------

Any kind of feedback/criticism would be greatly appreciated (software design,
documentation, improvement ideas, spelling mistakes, etc...).

If you'd like to see some features or algorithms implemented in
[Surprise](http://surpriselib.com), please let us know! Some of the current
ideas are:

- Bayesian PMF
- RBM for CF

Please feel free to contribute (see
[guidelines](https://github.com/NicolasHug/Surprise/blob/master/CONTRIBUTING.md))
and send pull requests!
