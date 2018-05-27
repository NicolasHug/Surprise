"""Run all the examples (except for benchmark.py). Just used as some kind of
functional test... If no warning / errors is output, it should be fine.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import sys
import os
import warnings

# redirect stout to /dev/null to avoid printing
sys.stdout = open(os.devnull, 'w')

from building_custom_algorithms import mean_rating_user_item
from building_custom_algorithms import most_basic_algorithm2
from building_custom_algorithms import most_basic_algorithm
from building_custom_algorithms import with_baselines_or_sim
import load_custom_dataset_predefined_folds
import load_from_dataframe
import baselines_conf
import generate_grid_search_cv_results_example
import load_from_dataframe
import train_test_split
import basic_usage
import grid_search_usage
import serialize_algorithm
import use_cross_validation_iterators
import k_nearest_neighbors
import similarity_conf
import precision_recall_at_k
import split_data_for_unbiased_estimation
import evaluate_on_trainset
import load_custom_dataset
import predict_ratings
import top_n_recommendations
