"""Run all the examples (except for benchmark.py). Just used as some kind of
functional test... If no warning / errors is output, it should be fine.
"""

# flake8: noqa

import os
import sys
import warnings

# redirect stout to /dev/null to avoid printing
sys.stdout = open(os.devnull, "w")

import baselines_conf
import basic_usage
import evaluate_on_trainset
import generate_grid_search_cv_results_example
import grid_search_usage
import k_nearest_neighbors
import load_custom_dataset
import load_custom_dataset_predefined_folds
import load_from_dataframe
import precision_recall_at_k
import predict_ratings
import serialize_algorithm
import similarity_conf
import split_data_for_unbiased_estimation
import top_n_recommendations
import train_test_split
import use_cross_validation_iterators
from building_custom_algorithms import (
    mean_rating_user_item,
    most_basic_algorithm,
    most_basic_algorithm2,
    with_baselines_or_sim,
)
