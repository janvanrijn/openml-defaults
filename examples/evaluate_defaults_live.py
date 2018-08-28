import argparse
import collections
import copy
import matplotlib.pyplot as plt
from collections import defaultdict

import openml
import openmldefaults
import os
import pandas as pd
import warnings

from examples.assemble_results import run as assemble_results

# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/'
                                'surrogate__adaboost__predictive_accuracy__c8.arff')
    parser.add_argument('--flip_performances', action='store_true', default=True)
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--max_num_defaults', type=int, default=None)
    parser.add_argument('--cv_iterations', type=int, default=10)
    parser.add_argument('--input_dir', type=str, default=os.path.expanduser('~') + '/habanero_experiments/openml-defaults')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/habanero_experiments/openml-defaults')
    parser.add_argument('--vs_strategy', type=str, default='greedy')
    parser.add_argument('--dir_structure', type=str, nargs='+',
                        default=['strategy_name', 'configuration_specification', 'task_id'])
    return parser.parse_args()


def plot(df, y_label, output_file):
    unique_ndefaults = getattr(df, 'n_defaults').unique()
    unique_strategies = df.strategy_name.unique()
    n_figs = len(unique_ndefaults)
    fig = plt.figure(figsize=(4*n_figs, 6))
    axes = [fig.add_subplot(1, n_figs, i) for i in range(1, n_figs + 1)]
    for i, num_defaults in enumerate(sorted(unique_ndefaults)):
        strategy_scores = []
        strategy_names = []
        for strategy in unique_strategies:
            df_fixedstrategy = df.loc[(df['strategy_name'] == strategy) &
                                      (df['n_defaults'] == num_defaults)]
            current_scores = df_fixedstrategy['evaluation'].tolist()
            strategy_scores.append(current_scores)
            strategy_names.append("%s (n=%d)" %(strategy, len(current_scores)))
            assert(len(df_fixedstrategy) <= 100) # depends on study
        axes[i].boxplot(strategy_scores)
        axes[i].set_xticklabels(strategy_names, rotation=45, ha='right')
        axes[i].set_title(str(num_defaults) + ' defaults')
    axes[0].set_ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(openmldefaults.utils.get_time(), 'saved to', output_file)


def normalize_scores(df, task_minscore, task_maxscore):
    def normalize(row):
        eval = row['evaluation']
        min_score = task_minscore[row['task_id']]
        max_score = task_maxscore[row['task_id']]
        if min_score != max_score:
            return (eval - min_score) / (max_score - min_score)
        else:
            return min_score

    df = copy.deepcopy(df)
    df['evaluation'] = df.apply(lambda row: normalize(row), axis=1)
    return df


def run():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        raise ValueError('Could not locate input directory: %s' % args.input_dir)

    dataset_name = os.path.basename(args.dataset_path)
    strategies_dir = os.path.join(args.input_dir, dataset_name, 'live_random_search')
    if not os.path.isdir(strategies_dir):
        raise ValueError('Could not find strategies directory: %s' % strategies_dir)
    results_file = os.path.join(strategies_dir, 'results.csv')
    if not os.path.isfile(results_file):
        print(openmldefaults.utils.get_time(), 'Generating results.csv, this can take a while')
        assemble_results(args.input_dir, args.dataset_path, args.dir_structure)
    print(openmldefaults.utils.get_time(), 'results.csv available.')
    df = pd.read_csv(filepath_or_buffer=results_file, sep=',')
    meta_data = openmldefaults.utils.get_dataset_metadata(args.dataset_path)

    df['c_type'] = df['configuration_specification'].apply(lambda x: int(x.split('_')[0][1:]))
    df['n_defaults'] = df['configuration_specification'].apply(lambda x: int(x.split('_')[1][1:]))
    df = df.groupby(['strategy_name', 'task_id', 'n_defaults']).mean().reset_index()

    df = df.loc[df['c_type'] == args.resized_grid_size]
    task_minscores = dict()
    task_maxscores = dict()
    for task_id in getattr(df, 'task_id').unique():
        df_task = df.loc[df['task_id'] == task_id]
        task_min = df_task.evaluation.min()
        task_max = df_task.evaluation.max()

        task_minscores[task_id] = task_min
        task_maxscores[task_id] = task_max

    outputfile_vanilla = os.path.join(args.output_dir, "%s_live.png" % dataset_name)
    plot(df, meta_data['scoring'], outputfile_vanilla)

    df_normalized = normalize_scores(df, task_minscores, task_maxscores)
    outputfile_normalized = os.path.join(args.output_dir, "%s_live__normalized.png" % dataset_name)
    plot(df_normalized, meta_data['scoring'], outputfile_normalized)


if __name__ == '__main__':
    run()
