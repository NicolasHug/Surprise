Refactoring of how we handle datasets
=====================================

What others are doing
---------------------

Python-recsys: https://github.com/ocelma/python-recsys
The API is pretty cool because you can do stuff like

svd.load_data(filename='./data/movielens/ratings.dat',
            sep='::',
            format={'col':0, 'row':1, 'value':2, 'ids': int})
or

data = Data()
format = {'col':0, 'row':1, 'value':2, 'ids': 'int'}
data.load(filename, sep='::', format=format)

which is a lot more user friendly than what we're doing right now.

As for implementation, the ratings are a list of tuples (r, u, i) which is
passed to a thirdparty library for matrix construction.



Crab:
Dataset handling seems to be completely hardcoded (looks like user defined
datasets are not supported).
When loading a dataset, ratings are stored in a dict with following format:
     {userID1:{itemID1:preference1, itemID2:preference2},
      userID2:{itemID1:preference3, itemID3:preference4}}

It is then converted to a 2Dnumpy array in MatrixPreferenceDataModel. Null
values are np.nan (not Zero)

Note: metric calculation is pretty easy and neat:
return np.sqrt((np.sum((y_pred - y_real) ** 2)) / y_real.shape[0])

Usage: (checkout http://muricoca.github.io/crab/tutorial.html for usage)
* load a dataset (hardcoded)
* build a model from it
* build a similarity object from the model (pearson, cos (I guess?), ...)
* build a recommender engine from both model and similarity
