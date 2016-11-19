[![GitHub version](https://badge.fury.io/gh/Niourf%2Frecsys.svg)](https://badge.fury.io/gh/Niourf%2Frecsys)
[![Documentation Status](https://readthedocs.org/projects/recsys/badge/?version=latest)](http://recsys.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/Niourf/RecSys.svg?branch=master)](https://travis-ci.org/Niourf/RecSys)
[![python_versions](https://img.shields.io/badge/python-2.7%2C%203.4%2C%203.5-blue.svg)]
(https://pypi.python.org/pypi/recsys/)
[![license](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://github.com/Niourf/RecSys/blob/master/LICENSE.md)


RecSys
======

Overview
--------

[RecSys](https://github.com/Niourf/RecSys) is an open source Python library
that provides with tools to build and evaluate the performance of many
recommender system prediction algorithms. Its goal is to make life easy(-ier)
for reseachers and students who want to play around with new recommender
algorithm ideas.

[RecSys](https://github.com/Niourf/RecSys) **was designed with the following
purposes in mind**:

- Give the user perfect control over his experiments. To this end, a strong
  emphasis is laid on
  [documentation](http://recsys.readthedocs.io/en/latest/index.html), which we
  have tried to make as clear and precise as possible by pointing out every
  details of the algorithms.
- Alleviate the pain of [Dataset
  handling](http://recsys.readthedocs.io/en/latest/getting_started.html#load-a-custom-dataset).
  Users can use both *built-in* datasets
  ([Movielens](http://grouplens.org/datasets/movielens/),
  [Jester](http://eigentaste.berkeley.edu/dataset/)), and their own *custom* datasets.
- Provide with various ready-to-use [prediction
  algorithms](http://recsys.readthedocs.io/en/latest/prediction_algorithms_package.html) (Neighborhood approaches, SVD, SVD++...)
- Make it easy to implement [new algorithm
  ideas](http://recsys.readthedocs.io/en/latest/building_custom_algo.html).
- Provide with tools to [evaluate](http://recsys.readthedocs.io/en/latest/evaluate.html),
  [analyse](http://nbviewer.jupyter.org/github/Niourf/RecSys/tree/master/examples/notebooks/KNNBasic_analysis.ipynb/)
  and
  [compare](http://nbviewer.jupyter.org/github/Niourf/RecSys/tree/master/examples/notebooks/Compare.ipynb/)
  the algorithms performance. Cross-validation procedures can be run very easily.

Installation / Usage
--------------------

To install from [PyPI](https://pypi.python.org/pypi/recsys/), use pip (you'll
need numpy):

    $ pip install recsys

Or clone the repo and build from the sources (you'll need Cython and numpy
first):

    $ git clone https://github.com/Niourf/recsys.git
    $ python setup.py install

Example
-------

Here is a simple example showing how one can (down)load a dataset, split it for
3-folds cross-validation,and compute the MAE and RMSE of the SVD algorithm.

```python
from recsys import SVD
from recsys import Dataset
from recsys import evaluate


# Load the movielens-100k dataset (it will be downloaded if needed),
# and split it into 3 folds for cross-validation.
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)

# We'll use the famous SVD algorithm.
algo = SVD()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print(perf['RMSE'])
print(perf['MAE'])
```

**Output**:

```
--------------------
fold 0
RMSE: 0.9461
MAE: 0.7471
--------------------
fold 1
RMSE: 0.9485
MAE: 0.7481
--------------------
fold 2
RMSE: 0.9373
MAE: 0.7389
--------------------
mean RMSE : 0.9440
mean MAE : 0.7447
[0.94610849207651793, 0.94851906980098399, 0.93725513525972337]
[0.74705780800352328, 0.74810449832136583, 0.73891237929484566]
```

Documentation, Getting Started
------------------------------

The documentation with many other usage examples is available
[online](http://recsys.readthedocs.io/en/latest/index.html) on ReadTheDocs.

License
-------

This project is licensed under the GPLv3 license - see the LICENSE.md file for
details.

Acknowledgements:
----------------

- [Pierre-François Gimenez](https://github.com/PFgimenez), for his valuable
  insights on software design.

Contributing
------------

Any kind of feedback/criticism would be greatly appreciated (software design,
documentation, improvement ideas, spelling mistakes, etc...). Please feel free
to contribute and send pull requests!
