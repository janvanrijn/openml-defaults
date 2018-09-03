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
                        default=os.path.expanduser('~') + '/data/openml-defaults/surrogate__libsvm_svc__predictive_accuracy__c8.arff')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--input_file', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults/20180826/libsvm_svc.csv')
    return parser.parse_args()


def plot(df, y_label, output_file):
    unique_strategies = df.strategy_name.unique()
    n_strategies = len(unique_strategies)
    fig = plt.figure(figsize=(n_strategies, 6))
    axes = fig.add_subplot(1, 1, 1)
    #for i, num_defaults in enumerate(sorted(unique_ndefaults)):
    strategy_scores = []
    strategy_names = []
    # TODO: sort strategies by (??) median performance?
    # TODO: rename strategies to "1 default", "2 defaults", "random search 1 itt", etc
    for strategy in unique_strategies:
        df_fixedstrategy = df.loc[(df['strategy_name'] == strategy)]
        current_scores = df_fixedstrategy['evaluation'].tolist()
        strategy_scores.append(current_scores)
        strategy_names.append("%s (n=%d)" %(strategy, len(current_scores)))
        assert(len(df_fixedstrategy) <= 100) # depends on study
    axes.boxplot(strategy_scores)
    axes.set_xticklabels(strategy_names, rotation=45, ha='right')
    axes.set_ylabel(y_label)
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
    if not os.path.isfile(args.input_file):
        raise ValueError('Could not locate input file: %s' % args.input_file)

    dataset_name = os.path.basename(args.dataset_path)
    output_dir = os.path.dirname(args.input_file)
    df = pd.read_csv(filepath_or_buffer=args.input_file, sep=',')
    meta_data = openmldefaults.utils.get_dataset_metadata(args.dataset_path)

    df['n_defaults'] = df['strategy_name'].apply(lambda x: int(x.split('__')[1]))
    df = df.groupby(['strategy_name', 'task_id']).mean().reset_index()

    df = df.loc[df['configuration_specification'] == args.resized_grid_size]
    task_minscores = dict()
    task_maxscores = dict()
    for task_id in getattr(df, 'task_id').unique():
        df_task = df.loc[df['task_id'] == task_id]
        task_min = df_task.evaluation.min()
        task_max = df_task.evaluation.max()

        task_minscores[task_id] = task_min
        task_maxscores[task_id] = task_max

    outputfile_vanilla = os.path.join(output_dir, "%s_live.png" % dataset_name)
    plot(df, meta_data['scoring'], outputfile_vanilla)

    df_normalized = normalize_scores(df, task_minscores, task_maxscores)
    outputfile_normalized = os.path.join(output_dir, "%s_live__normalized.png" % dataset_name)
    plot(df_normalized, meta_data['scoring'], outputfile_normalized)


if __name__ == '__main__':
    run()
