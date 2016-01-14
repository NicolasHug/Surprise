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
evaluate(algo, data, metrics=['RMSE', 'MAE', ...])


# dataset construction
(1) data = Dataset(name='ml-100k') or
(2) data = Dataset(ratings_file='path_to_file', format=format) or
(3) data = Dataset(train_file='path_to_file', test_file='path_to_file', format=format)
(4) data = Dataset(folds_files=folds_file, format=format)

Note: we need to consider having two Dataset subclasses: one when the folds are
already known (3 and 4) and one when user provides only a single ratings file
(1 and 2).

format is a string indicating the format of a line in a rating file:
format='user :: item :: rating :: timestamp' or
format='item | rating; user , timestamp' <-- maybe we won't handle such an
absurd format...
UPDATE: ok so for now user has to explicitely give the sep, and it has to be
unique.

a dataset object would have a 'raw_folds' attribute, which would be a
generator:
((raw_test_1, raw_train_1), (raw_test_2, raw_train_2), ...)
raw_test_x and raw_train_x are lists of ratings that will be converted to a
proper data structure suitable for algorithms when the folds() method is called.

for (3) and (4), the 'raw_folds' attribute is built immediately, else its
construction is deferred to the makeCV method.

@property
def raw_folds(self):
    ## for 3 and 4
    for trainfile, testfile in self.folds_files:
        raw_trainset = self.read_ratings(open(trainfile))
        raw_testset = self.read_ratings(open(testfile))
        yield raw_trainset, raw_testset
    ## for 1 and 2
    combine previou call to makeCV() and yield appropriate raw_****set


@property
def folds(self):
    for raw_trainset, raw_testset in raw_folds:
        trainset, testset = self.convert_for_algo(raw_trainset, raw_testset)
        yield trainset, testset

convert_for_algo (name to be changed) converts a raw dataset (which is just a
list of ratings) to a structure suitable for algorithms: dictionaries rm, ur
and ir (change names as well)

def makeCV(self, n_folds=5):
    # this basically constructs something, which will be used by the raw_folds
    generator


NOTE: as raw_folds and folds are generator, files are only opened and read when
needed. It might be good idea to check at least if they all exist at the
beggining, so that the program does not crash on the 10th fold...

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
