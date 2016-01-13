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


What we want to do
------------------

the workflow could look like this (names of parameters and methods are subject
to change):

data = Dataset(...)
algo = KNNBaseline(user_based=True, options=options, sim=sim)
data.makeCV(n_folds=5) # optional and depending how data was constructed
evaluate(algo, data) # this would give us RMSE, MAE, etc...


# dataset construction
(1) data = Dataset(name='ml-100k') or
(2) data = Dataset(ratings_file='path_to_file', format=format) or
(3) data = Dataset(train_file='path_to_file', test_file='path_to_file', format=format)
(4) data = Dataset(folds_files=folds_file, format=format)

format is a string indicating the format of a line in a rating file:
format='user :: item :: rating :: timestamp' or
format='item | rating; user , timestamp' <-- maybe we won't handle such an
absurd format...

a dataset object would have a 'folds' attribute:
[(test_1, train_1), (test_2, train_2), ...]
test_x and train_x are lists of ratings.

for (3) and (4), the 'folds' attribute is
built immediately, else its construction is deferred to the makeCV method.

# algorithm construction
algo = KNNBaseline(user_based=True, options=options, sim=sim)

options would be a dict of all the options related to the algorithm. For
example for KNNBaseline, keys would be 'method', and depending on the method
'reg_u', 'reg_i'... or 'n_epochs'... etc. We should be careful not to have
conflicting parameters names (clearly n_epoch could be one of them).

sim (optional) is as well a dict with the name of the similarity measure to use, and
related options, such as the minimum number of items in common, shrinkage
parameters and stuff...
We chose to not to include it in the 'options' parameters because sim
seems to be an entity of its own, but that might change.

The evaluate function could look like:
def evaluate(algo, data):
    for train_set, test_set in data.folds:
        algo.train(train_set)
        algo.test(test_set)

    ... aggregate results etc...
