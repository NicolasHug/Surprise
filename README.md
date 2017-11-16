[![GitHub version](https://badge.fury.io/gh/nicolashug%2FSurprise.svg)](https://badge.fury.io/gh/nicolashug%2FSurprise)
[![Documentation Status](https://readthedocs.org/projects/surprise/badge/?version=stable)](http://surprise.readthedocs.io/en/stable/?badge=stable)
[![Build Status](https://travis-ci.org/NicolasHug/Surprise.svg?branch=master)](https://travis-ci.org/NicolasHug/Surprise)
[![python versions](https://img.shields.io/badge/python-2.7%2C%203.5%2C%203.6-blue.svg)](http://surpriselib.com)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)


Surprise
========

Overview
--------

[Surprise](http://surpriselib.com) is a Python
[scikit](https://www.scipy.org/scikits.html) building and analyzing recommender
systems.

[Surprise](http://surpriselib.com) **was designed with the
following purposes in mind**:

- Give users perfect control over their experiments. To this end, a strong
  emphasis is laid on
  [documentation](http://surprise.readthedocs.io/en/stable/index.html), which we
  have tried to make as clear and precise as possible by pointing out every
  detail of the algorithms.
- Alleviate the pain of [Dataset
  handling](http://surprise.readthedocs.io/en/stable/getting_started.html#load-a-custom-dataset).
  Users can use both *built-in* datasets
  ([Movielens](http://grouplens.org/datasets/movielens/),
  [Jester](http://eigentaste.berkeley.edu/dataset/)), and their own *custom*
  datasets.
- Provide various ready-to-use [prediction
  algorithms](http://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html)
  such as [baseline
  algorithms](http://surprise.readthedocs.io/en/stable/basic_algorithms.html),
  [neighborhood
  methods](http://surprise.readthedocs.io/en/stable/knn_inspired.html), matrix
  factorization-based (
  [SVD](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD),
  [PMF](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#unbiased-note),
  [SVD++](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp),
  [NMF](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF)),
  and [many
  others](http://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html).
  Also, various [similarity
  measures](http://surprise.readthedocs.io/en/stable/similarities.html)
  (cosine, MSD, pearson...) are built-in.
- Make it easy to implement [new algorithm
  ideas](http://surprise.readthedocs.io/en/stable/building_custom_algo.html).
- Provide tools to [evaluate](http://surprise.readthedocs.io/en/stable/evaluate.html),
  [analyse](http://nbviewer.jupyter.org/github/NicolasHug/Surprise/tree/master/examples/notebooks/KNNBasic_analysis.ipynb/)
  and
  [compare](http://nbviewer.jupyter.org/github/NicolasHug/Surprise/blob/master/examples/notebooks/Compare.ipynb)
  the algorithms performance. Cross-validation procedures can be run very
  easily, as well as [exhaustive search over a set of
  parameters](http://surprise.readthedocs.io/en/stable/getting_started.html#tune-algorithm-parameters-with-gridsearch).


The name *SurPRISE* (roughly :) ) stands for Simple Python RecommendatIon
System Engine.


Getting started, example
------------------------

Here is a simple example showing how you can (down)load a dataset, split it for
3-folds cross-validation, and compute the MAE and RMSE of the
[SVD](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)
algorithm.

```python
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf


# Load the movielens-100k dataset (download it if needed),
# and split it into 3 folds for cross-validation.
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)
```

**Output**:

```
Evaluating RMSE, MAE of algorithm SVD.

        Fold 1  Fold 2  Fold 3  Mean
MAE     0.7475  0.7447  0.7425  0.7449
RMSE    0.9461  0.9436  0.9425  0.9441
```

[Surprise](http://surpriselib.com) can do **much** more (e.g,
[GridSearch](http://surprise.readthedocs.io/en/stable/getting_started.html#tune-algorithm-parameters-with-gridsearch))!
You'll find [more usage
examples](http://surprise.readthedocs.io/en/stable/getting_started.html) in the
[documentation ](http://surprise.readthedocs.io/en/stable/index.html).


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
| [NormalPredictor](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor) | 1.5228 | 1.2242 |     4    |
| [BaselineOnly](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly)    |  .9445 |  .7488 |    5    |
| [KNNBasic](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic)        |  .9789 |  .7732 |    27    |
| [KNNWithMeans](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans)    |  .9514 |  .7500 |    30    |
| [KNNBaseline](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline)     |  .9306 |  .7334 |    44    |
| [SVD](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)             |  .9364 |  .7381 |    46    |
| [SVD++](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp)             |  .9200 |  .7253 |    31min    |
| [NMF](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF)             |  .9634 |  .7572 |    55    |
| [Slope One](http://surprise.readthedocs.io/en/stable/slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne)             |  .9454 |  .7430 |    25    |
| [Co clustering](http://surprise.readthedocs.io/en/stable/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering)             |  .9678 |  .7579 |    15    |


|  [Movielens 1M](http://grouplens.org/datasets/movielens/1m) |  RMSE  |   MAE  | Time (min) |
|-----------------|:------:|:------:|:--------:|
| [NormalPredictor](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor) | 1.5037 | 1.2051 |     < 1    |
| [BaselineOnly](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly)    |  .9086 | .7194 |    < 1    |
| [KNNBasic](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic)        |  .9207 |  .7250 |    22    |
| [KNNWithMeans](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans)    |  .9292 |  .7386 |    22    |
| [KNNBaseline](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline)     |  .8949 | .7063 |    44    |
| [SVD](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)             |  .8738 |  .6858 |    7    |
| [NMF](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF)             |  .9155 |  .7232 |    9    |
| [Slope One](http://surprise.readthedocs.io/en/stable/slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne)             |  .9065 |  .7144 |    8    |
| [Co clustering](http://surprise.readthedocs.io/en/stable/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering)             |  .9155 |  .7174 |    2    |

Installation
------------

The easiest way is to use pip (you'll need [numpy](http://www.numpy.org/)):

    $ pip install numpy
    $ pip install scikit-surprise

If you use conda, you can install the package following [these
guidelines](https://conda.io/docs/user-guide/getting-started.html#installing-a-package-with-pip)
(still using pip).

For the latest version, you can also clone the repo and build the source
(you'll first need [Cython](http://cython.org/) and
[numpy](http://www.numpy.org/)):

    $ pip install numpy cython
    $ git clone https://github.com/NicolasHug/surprise.git
    $ cd surprise
    $ python setup.py install

License
-------

This project is licensed under the [BSD
3-Clause](https://opensource.org/licenses/BSD-3-Clause) license, so it can be
used for pretty much everything, including commercial applications. Please let
us know how [Surprise](http://surpriselib.com) is useful to you!

Here is a Bibtex entry if you ever need to cite Surprise in a research paper
(please keep us posted, we would love to know if Surprise was helpful to you):

    @Misc{Surprise,
    author =   {Hug, Nicolas},
    title =    { {S}urprise, a {P}ython library for recommender systems},
    howpublished = {\url{http://surpriselib.com}},
    year = {2017}
    }

Contributors
------------

The following persons have contributed to [Surprise](http://surpriselib.com):

Charles-Emmanuel Dias, Lukas Galke, Pierre-François Gimenez, Nicolas Hug,
Hengji Liu,  Maher Malaeb, Naturale0, nju-luke, Skywhat, Mike Lee Williams,
Chenchen Xu.

Thanks a lot :) !

Contributing, feedback, contact
-------------------------------

Any kind of feedback/criticism would be greatly appreciated (software design,
documentation, improvement ideas, spelling mistakes, etc...).

If you'd like to see some features or algorithms implemented in
[Surprise](http://surpriselib.com), please let us know!

Please feel free to contribute (see
[guidelines](https://github.com/NicolasHug/Surprise/blob/master/.github/CONTRIBUTING.md))
and send pull requests!

For bugs, issues or questions about [Surprise](http://surpriselib.com), you can
use the GitHub [project page](https://github.com/NicolasHug/Surprise) (please
don't send me emails as there would be no record for future users).
