#!/usr/bin/env python

import argparse
import os
import random as rd
import shutil
import sys

import numpy as np

import surprise.dataset as dataset
from surprise import __version__
from surprise.builtin_datasets import get_dataset_dir
from surprise.dataset import Dataset
from surprise.model_selection import cross_validate, KFold, PredefinedKFold

from surprise.prediction_algorithms import (
    BaselineOnly,
    CoClustering,
    KNNBaseline,
    KNNBasic,
    KNNWithMeans,
    NMF,
    NormalPredictor,
    SlopeOne,
    SVD,
    SVDpp,
)
from surprise.reader import Reader  # noqa


def main():
    class MyParser(argparse.ArgumentParser):
        """A parser which prints the help message when an error occurs. Taken from
        https://stackoverflow.com/questions/4042452/display-help-message-with-python-argparse-when-script-is-called-without-any-argu."""  # noqa

        def error(self, message):
            sys.stderr.write("error: %s\n" % message)
            self.print_help()
            sys.exit(2)

    parser = MyParser(
        description="Evaluate the performance of a rating prediction "
        + "algorithm "
        + "on a given dataset using cross validation. You can use a built-in "
        + "or a custom dataset, and you can choose to automatically split the "
        + "dataset into folds, or manually specify train and test files. "
        + "Please refer to the documentation page "
        + "(https://surprise.readthedocs.io/) for more details.",
        epilog="""Example:\n
        surprise -algo SVD -params "{'n_epochs': 5, 'verbose': True}"
        -load-builtin ml-100k -n-folds 3""",
    )

    algo_choices = {
        "NormalPredictor": NormalPredictor,
        "BaselineOnly": BaselineOnly,
        "KNNBasic": KNNBasic,
        "KNNBaseline": KNNBaseline,
        "KNNWithMeans": KNNWithMeans,
        "SVD": SVD,
        "SVDpp": SVDpp,
        "NMF": NMF,
        "SlopeOne": SlopeOne,
        "CoClustering": CoClustering,
    }

    parser.add_argument(
        "-algo",
        type=str,
        choices=algo_choices,
        help="The prediction algorithm to use. "
        + "Allowed values are "
        + ", ".join(algo_choices.keys())
        + ".",
        metavar="<prediction algorithm>",
    )

    parser.add_argument(
        "-params",
        type=str,
        metavar="<algorithm parameters>",
        default="{}",
        help="A kwargs dictionary that contains all the "
        + "algorithm parameters."
        + "Example: \"{'n_epochs': 10}\".",
    )

    parser.add_argument(
        "-load-builtin",
        type=str,
        dest="load_builtin",
        metavar="<dataset name>",
        default="ml-100k",
        help="The name of the built-in dataset to use."
        + "Allowed values are "
        + ", ".join(dataset.BUILTIN_DATASETS.keys())
        + ". Default is ml-100k.",
    )

    parser.add_argument(
        "-load-custom",
        type=str,
        dest="load_custom",
        metavar="<file path>",
        default=None,
        help="A file path to custom dataset to use. "
        + "Ignored if "
        + "-loadbuiltin is set. The -reader parameter needs "
        + "to be set.",
    )

    parser.add_argument(
        "-folds-files",
        type=str,
        dest="folds_files",
        metavar="<train1 test1 train2 test2... >",
        default=None,
        help="A list of custom train and test files. "
        + "Ignored if -load-builtin or -load-custom is set. "
        "The -reader parameter needs to be set.",
    )

    parser.add_argument(
        "-reader",
        type=str,
        metavar="<reader>",
        default=None,
        help="A Reader to read the custom dataset. Example: "
        + "\"Reader(line_format='user item rating timestamp',"
        + " sep='\\t')\"",
    )

    parser.add_argument(
        "-n-folds",
        type=int,
        dest="n_folds",
        metavar="<number of folds>",
        default=5,
        help="The number of folds for cross-validation. " + "Default is 5.",
    )

    parser.add_argument(
        "-seed",
        type=int,
        metavar="<random seed>",
        default=None,
        help="The seed to use for RNG. " + "Default is the current system time.",
    )

    parser.add_argument(
        "--with-dump",
        dest="with_dump",
        action="store_true",
        help="Dump the algorithm "
        + "results in a file (one file per fold). "
        + "Default is False.",
    )

    parser.add_argument(
        "-dump-dir",
        dest="dump_dir",
        type=str,
        metavar="<dir>",
        default=None,
        help="Where to dump the files. Ignored if "
        + "with-dump is not set. Default is "
        + os.path.join(get_dataset_dir(), "dumps/"),
    )

    parser.add_argument(
        "--clean",
        dest="clean",
        action="store_true",
        help="Remove the " + get_dataset_dir() + " directory and exit.",
    )

    parser.add_argument("-v", "--version", action="version", version=__version__)

    args = parser.parse_args()

    if args.clean:
        folder = get_dataset_dir()
        shutil.rmtree(folder)
        print("Removed", folder)
        exit()

    # setup RNG
    rd.seed(args.seed)
    np.random.seed(args.seed)

    # setup algorithm
    params = eval(args.params)
    if args.algo is None:
        parser.error("No algorithm was specified.")
    algo = algo_choices[args.algo](**params)

    # setup dataset
    if args.load_custom is not None:  # load custom and split
        if args.reader is None:
            parser.error("-reader parameter is needed.")
        reader = eval(args.reader)
        data = Dataset.load_from_file(args.load_custom, reader=reader)
        cv = KFold(n_splits=args.n_folds, random_state=args.seed)

    elif args.folds_files is not None:  # load from files
        if args.reader is None:
            parser.error("-reader parameter is needed.")
        reader = eval(args.reader)
        folds_files = args.folds_files.split()
        folds_files = [
            (folds_files[i], folds_files[i + 1])
            for i in range(0, len(folds_files) - 1, 2)
        ]
        data = Dataset.load_from_folds(folds_files=folds_files, reader=reader)
        cv = PredefinedKFold()

    else:  # load builtin dataset and split
        data = Dataset.load_builtin(args.load_builtin)
        cv = KFold(n_splits=args.n_folds, random_state=args.seed)

    cross_validate(algo, data, cv=cv, verbose=True)


if __name__ == "__main__":
    main()
