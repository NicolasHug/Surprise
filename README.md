[![GitHub version](https://badge.fury.io/gh/nicolashug%2FSurprise.svg)](https://badge.fury.io/gh/nicolashug%2FSurprise)
[![Documentation Status](https://readthedocs.org/projects/surprise/badge/?version=stable)](https://surprise.readthedocs.io/en/stable/?badge=stable)
[![python versions](https://img.shields.io/badge/python-3.8+-blue.svg)](https://surpriselib.com)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02174/status.svg)](https://doi.org/10.21105/joss.02174)

[![logo](./logo_black.svg)](https://surpriselib.com)

Overview
--------

[Surprise](https://surpriselib.com) is a Python
[scikit](https://projects.scipy.org/scikits.html) for building and analyzing
recommender systems that deal with explicit rating data.

[Surprise](https://surpriselib.com) **was designed with the
following purposes in mind**:

- Give users perfect control over their experiments. To this end, a strong
  emphasis is laid on
  [documentation](https://surprise.readthedocs.io/en/stable/index.html), which we
  have tried to make as clear and precise as possible by pointing out every
  detail of the algorithms.
- Alleviate the pain of [Dataset
  handling](https://surprise.readthedocs.io/en/stable/getting_started.html#load-a-custom-dataset).
  Users can use both *built-in* datasets
  ([Movielens](https://grouplens.org/datasets/movielens/),
  [Jester](https://eigentaste.berkeley.edu/dataset/)), and their own *custom*
  datasets.
- Provide various ready-to-use [prediction
  algorithms](https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html)
  such as [baseline
  algorithms](https://surprise.readthedocs.io/en/stable/basic_algorithms.html),
  [neighborhood
  methods](https://surprise.readthedocs.io/en/stable/knn_inspired.html), matrix
  factorization-based (
  [SVD](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD),
  [PMF](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#unbiased-note),
  [SVD++](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp),
  [NMF](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF)),
  and [many
  others](https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html).
  Also, various [similarity
  measures](https://surprise.readthedocs.io/en/stable/similarities.html)
  (cosine, MSD, pearson...) are built-in.
- Make it easy to implement [new algorithm
  ideas](https://surprise.readthedocs.io/en/stable/building_custom_algo.html).
- Provide tools to [evaluate](https://surprise.readthedocs.io/en/stable/model_selection.html),
  [analyse](https://nbviewer.jupyter.org/github/NicolasHug/Surprise/tree/master/examples/notebooks/KNNBasic_analysis.ipynb/)
  and
  [compare](https://nbviewer.jupyter.org/github/NicolasHug/Surprise/blob/master/examples/notebooks/Compare.ipynb)
  the algorithms' performance. Cross-validation procedures can be run very
  easily using powerful CV iterators (inspired by
  [scikit-learn](https://scikit-learn.org/) excellent tools), as well as
  [exhaustive search over a set of
  parameters](https://surprise.readthedocs.io/en/stable/getting_started.html#tune-algorithm-parameters-with-gridsearchcv).


The name *SurPRISE* (roughly :) ) stands for *Simple Python RecommendatIon
System Engine*.

Please note that surprise does not support implicit ratings or content-based
information.


Getting started, example
------------------------

Here is a simple example showing how you can (down)load a dataset, split it for
5-fold cross-validation, and compute the MAE and RMSE of the
[SVD](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)
algorithm.


```python
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# Use the famous SVD algorithm.
algo = SVD()

# Run 5-fold cross-validation and print results.
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```

**Output**:

```
Evaluating RMSE, MAE of algorithm SVD on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.9367  0.9355  0.9378  0.9377  0.9300  0.9355  0.0029  
MAE (testset)     0.7387  0.7371  0.7393  0.7397  0.7325  0.7375  0.0026  
Fit time          0.62    0.63    0.63    0.65    0.63    0.63    0.01    
Test time         0.11    0.11    0.14    0.14    0.14    0.13    0.02    
```

[Surprise](https://surpriselib.com) can do **much** more (e.g,
[GridSearchCV](https://surprise.readthedocs.io/en/stable/getting_started.html#tune-algorithm-parameters-with-gridsearchcv))!
You'll find [more usage
examples](https://surprise.readthedocs.io/en/stable/getting_started.html) in the
[documentation ](https://surprise.readthedocs.io/en/stable/index.html).


Benchmarks
----------

Here are the average RMSE, MAE and total execution time of various algorithms
(with their default parameters) on a 5-fold cross-validation procedure. The
datasets are the [Movielens](https://grouplens.org/datasets/movielens/) 100k and
1M datasets. The folds are the same for all the algorithms. All experiments are
run on a laptop with an intel i5 11th Gen 2.60GHz. The code
for generating these tables can be found in the [benchmark
example](https://github.com/NicolasHug/Surprise/tree/master/examples/benchmark.py).

| [Movielens 100k](http://grouplens.org/datasets/movielens/100k)                                                                         |   RMSE |   MAE | Time    |
|:---------------------------------------------------------------------------------------------------------------------------------------|-------:|------:|:--------|
| [SVD](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)      |  0.934 | 0.737 | 0:00:06 |
| [SVD++ (cache_ratings=False)](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp)  |  0.919 | 0.721 | 0:01:39 |
| [SVD++ (cache_ratings=True)](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp)  |  0.919 | 0.721 | 0:01:22 |
| [NMF](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF)      |  0.963 | 0.758 | 0:00:06 |
| [Slope One](http://surprise.readthedocs.io/en/stable/slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne)                 |  0.946 | 0.743 | 0:00:09 |
| [k-NN](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic)                        |  0.98  | 0.774 | 0:00:08 |
| [Centered k-NN](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans)           |  0.951 | 0.749 | 0:00:09 |
| [k-NN Baseline](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline)            |  0.931 | 0.733 | 0:00:13 |
| [Co-Clustering](http://surprise.readthedocs.io/en/stable/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering) |  0.963 | 0.753 | 0:00:06 |
| [Baseline](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly)   |  0.944 | 0.748 | 0:00:02 |
| [Random](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor)    |  1.518 | 1.219 | 0:00:01 |


| [Movielens 1M](https://grouplens.org/datasets/movielens/1m)                                                                             |   RMSE |   MAE | Time    |
|:----------------------------------------------------------------------------------------------------------------------------------------|-------:|------:|:--------|
| [SVD](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)      |  0.873 | 0.686 | 0:01:07 |
| [SVD++ (cache_ratings=False)](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp)  |  0.862 | 0.672 | 0:41:06 |
| [SVD++ (cache_ratings=True)](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp)  |  0.862 | 0.672 | 0:34:55 |
| [NMF](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF)      |  0.916 | 0.723 | 0:01:39 |
| [Slope One](http://surprise.readthedocs.io/en/stable/slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne)                 |  0.907 | 0.715 | 0:02:31 |
| [k-NN](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic)                        |  0.923 | 0.727 | 0:05:27 |
| [Centered k-NN](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans)           |  0.929 | 0.738 | 0:05:43 |
| [k-NN Baseline](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline)            |  0.895 | 0.706 | 0:05:55 |
| [Co-Clustering](http://surprise.readthedocs.io/en/stable/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering) |  0.915 | 0.717 | 0:00:31 |
| [Baseline](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly)   |  0.909 | 0.719 | 0:00:19 |
| [Random](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor)    |  1.504 | 1.206 | 0:00:19 |

Installation
------------

With pip (you'll need [numpy](https://www.numpy.org/), and a C compiler. Windows
users might prefer using conda):

    $ pip install numpy
    $ pip install scikit-surprise

With conda:

    $ conda install -c conda-forge scikit-surprise

For the latest version, you can also clone the repo and build the source
(you'll first need [Cython](https://cython.org/) and
[numpy](https://www.numpy.org/)):

    $ pip install numpy cython
    $ git clone https://github.com/NicolasHug/surprise.git
    $ cd surprise
    $ python setup.py install

License and reference
---------------------

This project is licensed under the [BSD
3-Clause](https://opensource.org/licenses/BSD-3-Clause) license, so it can be
used for pretty much everything, including commercial applications.

I'd love to know how Surprise is useful to you. Please don't hesitate to open
an issue and describe how you use it!

Please make sure to cite the
[paper](https://joss.theoj.org/papers/10.21105/joss.02174) if you use
Surprise for your research:

    @article{Hug2020,
      doi = {10.21105/joss.02174},
      url = {https://doi.org/10.21105/joss.02174},
      year = {2020},
      publisher = {The Open Journal},
      volume = {5},
      number = {52},
      pages = {2174},
      author = {Nicolas Hug},
      title = {Surprise: A Python library for recommender systems},
      journal = {Journal of Open Source Software}
    }

Contributors
------------

The following persons have contributed to [Surprise](https://surpriselib.com):

ashtou, Abhishek Bhatia, bobbyinfj, caoyi, Chieh-Han Chen,  Raphael-Dayan, Олег
Демиденко, Charles-Emmanuel Dias, dmamylin, Lauriane Ducasse, Marc Feger,
franckjay, Lukas Galke, Tim Gates, Pierre-François Gimenez, Zachary Glassman,
Jeff Hale, Nicolas Hug, Janniks, jyesawtellrickson, Doruk Kilitcioglu, Ravi Raju
Krishna, lapidshay, Hengji Liu, Ravi Makhija, Maher Malaeb, Manoj K, James
McNeilis, Naturale0, nju-luke, Pierre-Louis Pécheux, Jay Qi, Lucas Rebscher,
Craig Rodrigues, Skywhat, Hercules Smith, David Stevens, Vesna Tanko,
TrWestdoor, Victor Wang, Mike Lee Williams, Jay Wong, Chenchen Xu, YaoZh1918.

Thanks a lot :) !

Development Status
------------------

Starting from version 1.1.0 (September 2019), I will only maintain the package,
provide bugfixes, and perhaps sometimes perf improvements. I have less time to
dedicate to it now, so I'm unabe to consider new features.

For bugs, issues or questions about [Surprise](https://surpriselib.com), please
avoid sending me emails; I will most likely not be able to answer). Please use
the GitHub [project page](https://github.com/NicolasHug/Surprise) instead, so
that others can also benefit from it.
