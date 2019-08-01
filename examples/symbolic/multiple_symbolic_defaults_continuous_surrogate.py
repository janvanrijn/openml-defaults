import arff
import argparse
import ConfigSpace
import logging
import numpy as np
import openmlcontrib
import openmldefaults
import os
import pandas as pd
import pickle
import sklearn
import typing


def parse_args():
    metadata_svc = os.path.expanduser('~/projects/sklearn-bot/data/svc.arff')
    metadata_qualities = os.path.expanduser('~/projects/openml-python-contrib/data/metafeatures_openml100.arff')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/symbolic_defaults/')
    parser.add_argument('--task_idx', type=int, default=None)
    parser.add_argument('--metadata_performance_file', type=str, default=metadata_svc)
    parser.add_argument('--metadata_qualities_file', type=str, default=metadata_qualities)
    parser.add_argument('--search_qualities', type=str, nargs='+')
    parser.add_argument('--search_hyperparameters', type=str, nargs='+')
    parser.add_argument('--search_transform_fns', type=str, nargs='+')
    parser.add_argument('--classifier_name', type=str, default='svc', help='scikit-learn flow name')
    parser.add_argument('--search_space_identifier', type=str, default=None)
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--n_estimators', type=int, default=64)
    parser.add_argument('--task_id_column', default='task_id', type=str)
    parser.add_argument('--skip_row_check', action='store_true')
    return parser.parse_args()


def run(args):
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
    pass


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
