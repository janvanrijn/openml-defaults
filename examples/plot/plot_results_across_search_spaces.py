import argparse
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import openmldefaults
import os
import pandas as pd


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
STRICT_CHECK = False


def run(args):
    pd.options.mode.chained_assignment = 'raise'
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    folder_constraints = {
        2: ['max_predictive_accuracy'],
        3: ['32'],
        4: ['0']
    }
    folder_legend = {
        'folder_depth_0': 'search_space',
        'folder_depth_1': 'task_id',
        'folder_depth_2': 'strategy',
        'folder_depth_3': 'num_defaults_scheduled',
        'folder_depth_4': 'random_seed',
        'folder_depth_5': 'param_aggregate',
        'folder_depth_6': 'param_a3r_r',
        'folder_depth_7': 'param_normalize_base',
        'folder_depth_8': 'param_normalize_a3r',
    }

    result_total = None
    for budget in ALL_BUDGETS:
        result_budget, _ = openmldefaults.utils.results_from_folder_to_df(args.results_directory,
                                                                          args.n_defaults_in_file,
                                                                          budget,
                                                                          folder_constraints,
                                                                          False,
                                                                          False)
        result_budget['budget'] = budget
        if result_total is None:
            result_total = result_budget
        else:
            result_total = result_total.append(result_budget)
    expectation = EXPECTED_DATASETS * EXPECTED_SEARCH_SPACES * len(ALL_BUDGETS)
    if result_total.shape[0] < expectation:
        msg = 'Not enough results! Expected at least %d, got %d' % (expectation,
                                                                    result_total.shape[0])
        if STRICT_CHECK:
            raise ValueError(msg)
        else:
            logging.warning(msg)

    result_total = result_total.rename(index=str, columns=folder_legend)
    result_total[args.scoring] = result_total[args.scoring].astype(float)

    index_columns = ['search_space', 'task_id', 'strategy', 'random_seed', 'param_aggregate',
                     'param_a3r_r', 'param_normalize_base', 'param_normalize_a3r']
    openmldefaults.utils.check_budget_curves(result_total, index_columns, args.scoring, 'budget', 0.0, 1.0)

    for column in ['param_aggregate', 'param_a3r_r', 'param_normalize_base', 'param_normalize_a3r']:
        if len(result_total[column].unique()) > 2:
            raise ValueError('Expected at most 2 values for column %s. Got %s' % (column,
                                                                                  result_total[column].unique()))

    result_total_normalized = openmldefaults.utils.normalize_df_conditioned_on(result_total,
                                                                               'MinMaxScaler',
                                                                               args.scoring,
                                                                               'task_id')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # absolute plots
    sns.boxplot(x="budget", y=args.scoring, hue="search_space", data=result_total, ax=ax1)
    ax1.set_title("vanilla")
    sns.boxplot(x="budget", y=args.scoring, hue="search_space", data=result_total_normalized, ax=ax2)
    ax2.set_title("normalized (per task)")

    os.makedirs(args.output_directory, exist_ok=True)
    output_file = os.path.join(args.output_directory, 'results_across_search_spaces.png')
    # ax1.set_yscale('log')
    plt.savefig(output_file)
    logging.info('stored figure to %s' % output_file)


if __name__ == '__main__':
    run(parse_args())
