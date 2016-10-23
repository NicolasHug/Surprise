RecSys
======

version number: 0.0.2
author: Nicolas Hug

Overview
--------

RecSys is an open source Python package that provides with tools to build and
evaluate the performance of many recommender system prediction algorithms. Its
goal is to make life easy(-ier) for reseachers and students who want to play
around with new recommender algorithm ideas.

A strong emphasis is laid on
[documentation](http://recsys.readthedocs.io/en/latest/index.html), which we
have tried to make as clear and precise as possible by pointing out every
details of the algorithms, in order for the practitioner to have perfect
control over his experiments.

Installation / Usage
--------------------

To install use pip:

    $ pip install recsys


Or clone the repo:

    $ git clone https://github.com/Niourf/recsys.git
    $ python setup.py install

Example
-------

    In [1]: from recsys import KNNBasic, Dataset, evaluate
    In [2]: data = Dataset.load_builtin('ml-100k') # load the movielens dataset
		Dataset ml-100k could not be found. Do you want to download it? [Y/n]
		Trying to download dataset from http://files.grouplens.org/datasets/movielens/ml-100k.zip...
		Done! Dataset ml-100k has been saved to /home/nico/.recsys_data/ml-100k

    In [3]: data.split(n_folds=3) # split into 3 folds for cross-validation
    In [4]: algo = KNNBasic() # use a basic nearest neighbor approach
    In [5]: evaluate(algo, data) # evaluate performance of algorithm
    --------------------
    fold 0
    computing the similarity matrix...
    RMSE: 0.9904
    MAE: 0.7851
    FCP: 0.7099
    --------------------
    fold 1
    computing the similarity matrix...
    RMSE: 0.9833
    MAE: 0.7766
    FCP: 0.7113
    --------------------
    fold 2
    computing the similarity matrix...
    RMSE: 0.9889
    MAE: 0.7819
    FCP: 0.7138
    --------------------
    mean RMSE : 0.9875
    mean MAE : 0.7812
    mean FCP : 0.7117


Documentation, Getting Started
------------------------------

The documentation with many usage examples is available
[online](http://recsys.readthedocs.io/en/latest/index.html) on ReadTheDocs.

License
-------

This project is licensed under the GPLv3 license - see the LICENSE.md file for
details.
