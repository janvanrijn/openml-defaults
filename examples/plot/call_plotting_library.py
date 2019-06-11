import argparse
import collections
import logging
import openmldefaults
import os
import pandas as pd
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--results_directory', type=str, default=os.path.expanduser('~/experiments/openml-defaults/active_testing'))
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~/experiments/openml-defaults/active_testing'))
    parser.add_argument('--python_venv', type=str, default=os.path.expanduser('~/anaconda3/envs/openml-defaults/bin/python'))
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--classifier_name', type=str, default='text_classification')
    parser.add_argument('--n_defaults_in_file', type=int, default=384)
    return parser.parse_args()


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    script_ranks = os.path.expanduser('~/projects/plotting_scripts/scripts/plot_ranks_from_csv.py')
    script_perf = os.path.expanduser('~/projects/plotting_scripts/scripts/plot_test_performance_from_csv.py')
    folder_constraints = {
        2: ['384'],
        3: ['0'],
        4: ['sum'],
        5: ['64'],
        6: ['None'],
        7: ['None']
    }
    folder_legend = {
        'folder_depth_0': 'task_id',
        'folder_depth_1': 'strategy',
        'folder_depth_2': 'n_defaults_max',
        'folder_depth_3': 'random_seed',
        'folder_depth_4': 'param_aggregate',
        'folder_depth_5': 'param_a3r_r',
        'folder_depth_6': 'param_normalize_base',
        'folder_depth_7': 'param_normalize_a3r',
    }
    # folder_constraints = None
    results_directory = os.path.join(args.results_directory, args.classifier_name)
    result_files = openmldefaults.utils.misc._traverse_run_folders(results_directory,
                                                                   384, list(), folder_constraints, True)
    df = pd.DataFrame(result_files, columns=list(folder_legend.values()) + ['filename'])
    df['total_path'] = df.apply(lambda col: '%s/%s' % (results_directory, '/'.join(col.values)), axis=1)
    df['command_ranks'] = df.apply(lambda col: '%s %s %s' % (col['task_id'],
                                                             col['strategy'],
                                                             col['total_path']), axis=1)
    strategy_files = collections.defaultdict(list)
    for index, row in df.iterrows():
        strategy_files[row['strategy']].append(row['total_path'])
    command_perf = ''
    for strategy, files in strategy_files.items():
        command_perf += strategy + ' ' + ' '.join(files) + ' '

    parameters_ranks = '--save %s/ranks.pdf --logx --ylabel "Avg. rank" --xlabel "Runtime (seconds)"' % args.output_directory
    subprocess.call('%s %s %s %s' % (args.python_venv, script_ranks, ' '.join(df['command_ranks'].values), parameters_ranks),
                    shell=True)

    parameters_acc = '--save %s/accuracy.pdf --logx --ylabel "Accuracy" --xlabel "Runtime (seconds)"' % args.output_directory
    subprocess.call('%s %s %s %s' % (args.python_venv, script_perf, command_perf, parameters_acc), shell=True)


if __name__ == '__main__':
    run(parse_args())
