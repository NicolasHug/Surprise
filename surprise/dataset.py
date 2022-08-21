"""
The :mod:`dataset <surprise.dataset>` module defines the :class:`Dataset` class
and other subclasses which are used for managing datasets.

Users may use both *built-in* and user-defined datasets (see the
:ref:`getting_started` page for examples). Right now, three built-in datasets
are available:

* The `movielens-100k <https://grouplens.org/datasets/movielens/>`_ dataset.
* The `movielens-1m <https://grouplens.org/datasets/movielens/>`_ dataset.
* The `Jester <https://eigentaste.berkeley.edu/dataset/>`_ dataset 2.

Built-in datasets can all be loaded (or downloaded if you haven't already)
using the :meth:`Dataset.load_builtin` method.
Summary:

.. autosummary::
    :nosignatures:

    Dataset.load_builtin
    Dataset.load_from_file
    Dataset.load_from_folds
"""


import itertools
import os
import sys
from collections import defaultdict

from .builtin_datasets import BUILTIN_DATASETS, download_builtin_dataset

from .reader import Reader
from .trainset import Trainset


class Dataset:
    """Base class for loading datasets.

    Note that you should never instantiate the :class:`Dataset` class directly
    (same goes for its derived classes), but instead use one of the three
    available methods for loading datasets."""

    def __init__(self, reader):

        self.reader = reader

    @classmethod
    def load_builtin(cls, name="ml-100k", prompt=True):
        """Load a built-in dataset.

        If the dataset has not already been loaded, it will be downloaded and
        saved. You will have to split your dataset using the :meth:`split
        <DatasetAutoFolds.split>` method. See an example in the :ref:`User
        Guide <cross_validate_example>`.

        Args:
            name(:obj:`string`): The name of the built-in dataset to load.
                Accepted values are 'ml-100k', 'ml-1m', and 'jester'.
                Default is 'ml-100k'.
            prompt(:obj:`bool`): Prompt before downloading if dataset is not
                already on disk.
                Default is True.

        Returns:
            A :obj:`Dataset` object.

        Raises:
            ValueError: If the ``name`` parameter is incorrect.
        """

        try:
            dataset = BUILTIN_DATASETS[name]
        except KeyError:
            raise ValueError(
                "unknown dataset "
                + name
                + ". Accepted values are "
                + ", ".join(BUILTIN_DATASETS.keys())
                + "."
            )

        # if dataset does not exist, offer to download it
        if not os.path.isfile(dataset.path):
            answered = not prompt
            while not answered:
                print(
                    "Dataset " + name + " could not be found. Do you want "
                    "to download it? [Y/n] ",
                    end="",
                )
                choice = input().lower()

                if choice in ["yes", "y", "", "omg this is so nice of you!!"]:
                    answered = True
                elif choice in ["no", "n", "hell no why would i want that?!"]:
                    answered = True
                    print("Ok then, I'm out!")
                    sys.exit()

            download_builtin_dataset(name)

        reader = Reader(**dataset.reader_params)

        return cls.load_from_file(file_path=dataset.path, reader=reader)

    @classmethod
    def load_from_file(cls, file_path, reader):
        """Load a dataset from a (custom) file.

        Use this if you want to use a custom dataset and all of the ratings are
        stored in one file. You will have to split your dataset using the
        :meth:`split <DatasetAutoFolds.split>` method. See an example in the
        :ref:`User Guide <load_from_file_example>`.


        Args:
            file_path(:obj:`string`): The path to the file containing ratings.
            reader(:obj:`Reader <surprise.reader.Reader>`): A reader to read
                the file.
        """

        return DatasetAutoFolds(ratings_file=file_path, reader=reader)

    @classmethod
    def load_from_folds(cls, folds_files, reader):
        """Load a dataset where folds (for cross-validation) are predefined by
        some files.

        The purpose of this method is to cover a common use case where a
        dataset is already split into predefined folds, such as the
        movielens-100k dataset which defines files u1.base, u1.test, u2.base,
        u2.test, etc... It can also be used when you don't want to perform
        cross-validation but still want to specify your training and testing
        data (which comes down to 1-fold cross-validation anyway). See an
        example in the :ref:`User Guide <load_from_folds_example>`.


        Args:
            folds_files(:obj:`iterable` of :obj:`tuples`): The list of the
                folds. A fold is a tuple of the form ``(path_to_train_file,
                path_to_test_file)``.
            reader(:obj:`Reader <surprise.reader.Reader>`): A reader to read
                the files.

        """

        return DatasetUserFolds(folds_files=folds_files, reader=reader)

    @classmethod
    def load_from_df(cls, df, reader):
        """Load a dataset from a pandas dataframe.

        Use this if you want to use a custom dataset that is stored in a pandas
        dataframe. See the :ref:`User Guide<load_from_df_example>` for an
        example.

        Args:
            df(`Dataframe`): The dataframe containing the ratings. It must have
                three columns, corresponding to the user (raw) ids, the item
                (raw) ids, and the ratings, in this order.
            reader(:obj:`Reader <surprise.reader.Reader>`): A reader to read
                the file. Only the ``rating_scale`` field needs to be
                specified.
        """

        return DatasetAutoFolds(reader=reader, df=df)

    def read_ratings(self, file_name):
        """Return a list of ratings (user, item, rating, timestamp) read from
        file_name"""

        with open(os.path.expanduser(file_name)) as f:
            raw_ratings = [
                self.reader.parse_line(line)
                for line in itertools.islice(f, self.reader.skip_lines, None)
            ]
        return raw_ratings

    def construct_trainset(self, raw_trainset):

        raw2inner_id_users = {}
        raw2inner_id_items = {}

        current_u_index = 0
        current_i_index = 0

        ur = defaultdict(list)
        ir = defaultdict(list)

        # user raw id, item raw id, translated rating, time stamp
        for urid, irid, r, timestamp in raw_trainset:
            try:
                uid = raw2inner_id_users[urid]
            except KeyError:
                uid = current_u_index
                raw2inner_id_users[urid] = current_u_index
                current_u_index += 1
            try:
                iid = raw2inner_id_items[irid]
            except KeyError:
                iid = current_i_index
                raw2inner_id_items[irid] = current_i_index
                current_i_index += 1

            ur[uid].append((iid, r))
            ir[iid].append((uid, r))

        n_users = len(ur)  # number of users
        n_items = len(ir)  # number of items
        n_ratings = len(raw_trainset)

        trainset = Trainset(
            ur,
            ir,
            n_users,
            n_items,
            n_ratings,
            self.reader.rating_scale,
            raw2inner_id_users,
            raw2inner_id_items,
        )

        return trainset

    def construct_testset(self, raw_testset):

        return [(ruid, riid, r_ui_trans) for (ruid, riid, r_ui_trans, _) in raw_testset]


class DatasetUserFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are predefined."""

    def __init__(self, folds_files=None, reader=None):

        Dataset.__init__(self, reader)
        self.folds_files = folds_files

        # check that all files actually exist.
        for train_test_files in self.folds_files:
            for f in train_test_files:
                if not os.path.isfile(os.path.expanduser(f)):
                    raise ValueError("File " + str(f) + " does not exist.")


class DatasetAutoFolds(Dataset):
    """A derived class from :class:`Dataset` for which folds (for
    cross-validation) are not predefined. (Or for when there are no folds at
    all)."""

    def __init__(self, ratings_file=None, reader=None, df=None):

        Dataset.__init__(self, reader)
        self.has_been_split = False  # flag indicating if split() was called.

        if ratings_file is not None:
            self.ratings_file = ratings_file
            self.raw_ratings = self.read_ratings(self.ratings_file)
        elif df is not None:
            self.df = df
            self.raw_ratings = [
                (uid, iid, float(r), None)
                for (uid, iid, r) in self.df.itertuples(index=False)
            ]
        else:
            raise ValueError("Must specify ratings file or dataframe.")

    def build_full_trainset(self):
        """Do not split the dataset into folds and just return a trainset as
        is, built from the whole dataset.

        User can then query for predictions, as shown in the :ref:`User Guide
        <train_on_whole_trainset>`.

        Returns:
            The :class:`Trainset <surprise.Trainset>`.
        """

        return self.construct_trainset(self.raw_ratings)
