import argparse
import numpy as np
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import openmldefaults
import os
import pandas as pd
import typing


def parse_args():
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--results_directory', type=str, default=os.path.expanduser('~/habanero_experiments/openml-defaults/vanilla_vs_rs'))
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~/experiments/openml-defaults/vanilla_vs_rs'))
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--n_defaults_in_file', type=int, default=32)
    return parser.parse_args()


EXPECTED_DATASETS = 99
EXPECTED_SEARCH_SPACES = 3
ALL_BUDGETS = [1, 2, 4, 8, 16, 32]
STRICT_CHECK = True


def run(args):
    result_total = None
    folder_constraints = {
        3: ['0']
    }
    folder_legend = {
        'folder_depth_0': 'search_space',
        'folder_depth_1': 'task_id',
        'folder_depth_2': 'strategy',
        'folder_depth_3': 'random_seed',
        'folder_depth_4': 'param_aggregate',
        'folder_depth_5': 'param_a3r_r',
        'folder_depth_6': 'param_normalize_base',
        'folder_depth_7': 'param_normalize_a3r',
    }

    result_curves = None
    for budget in ALL_BUDGETS:
        result_budget, _ = openmldefaults.utils.results_from_folder_to_df(args.results_directory,
                                                                                      args.n_defaults_in_file,
                                                                                      budget,
                                                                                      folder_constraints,
                                                                                      False)
        if result_budget.shape[0] < EXPECTED_DATASETS * EXPECTED_SEARCH_SPACES:
            msg = 'Not enough results! Expected at least %d, got %d' % (EXPECTED_DATASETS * EXPECTED_SEARCH_SPACES,
                                                                        result_budget.shape[0])
            if STRICT_CHECK:
                raise ValueError(msg)
            else:
                logging.warning(msg)
        if result_total is None:
            result_total = result_budget
        else:
            result_total = result_total.append(result_budget)
    result_total = result_total.rename(index=str, columns=folder_legend)
    result_total[args.scoring] = result_total[args.scoring].astype(float)

    index_columns = ['search_space', 'task_id', 'strategy', 'random_seed', 'param_aggregate',
                     'param_a3r_r', 'param_normalize_base', 'param_normalize_a3r']
    openmldefaults.utils.check_budget_curves(result_total, index_columns, args.scoring, 'budget')
