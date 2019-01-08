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
    parser.add_argument('--results_directory', type=str, default=os.path.expanduser('~/habanero_experiments/openml-defaults'))
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~/experiments/openml-defaults'))
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--classifier_name', type=str, default='svc')
    parser.add_argument('--n_defaults_in_file', type=int, default=32)
    return parser.parse_args()


def filter_frame(df: pd.DataFrame, filters: typing.Dict):
    for filter_key, filter_value in filters.items():
        df = df.loc[df[filter_key] == filter_value]
    return df


def check_budget_curves(df, values_column):
    results_pivot = df.pivot_table(
        values=values_column,
        index=['folder_depth_0', 'folder_depth_1', 'folder_depth_2',
               'folder_depth_3', 'folder_depth_4', 'folder_depth_5'],
        columns='budget',
        aggfunc=np.mean)
    for index, row in results_pivot.iterrows():
        last_value = 0.0
        for column, value in row.iteritems():
            if value < last_value:
                logging.warning('Series not strictly improving at budget %d for %s' % (column, index))
            last_value = value


def run(args):
    usercpu_time = 'usercpu_time_millis'
    result_total = None
    folder_constraints = {
        2: ['sum'],
        3: ['1'],
        4: ['None'],
        5: ['None']
    }
    # folder_constraints = None
    results_directory = os.path.join(args.results_directory, args.classifier_name)
    for budget in [1, 2, 4, 8, 16, 32]:
        result_budget = openmldefaults.utils.results_from_folder_to_df(results_directory,
                                                                       args.n_defaults_in_file,
                                                                       budget,
                                                                       folder_constraints,
                                                                       False)
        result_budget['budget'] = budget
        print(result_budget.shape)
        if result_total is None:
            result_total = result_budget
        else:
            result_total = result_total.append(result_budget)

    result_total[args.scoring] = result_total[args.scoring].astype(float)
    result_total[usercpu_time] = result_total[usercpu_time].astype(float)

    # sanity check results
    check_budget_curves(result_total, args.scoring)
    check_budget_curves(result_total, usercpu_time)

    output_directory_full = os.path.join(args.output_directory, args.classifier_name)
    os.makedirs(output_directory_full, exist_ok=True)

    for normalize_base in result_total['folder_depth_4'].unique():
        for normalize_a3r in result_total['folder_depth_5'].unique():
            for a3r_r in result_total['folder_depth_3'].unique():
                for aggregate in result_total['folder_depth_2'].unique():
                    filters = {
                        'folder_depth_4': normalize_base,
                        'folder_depth_5': normalize_a3r,
                        'folder_depth_3': a3r_r,
                        'folder_depth_2': aggregate,
                    }
                    title = 'normalize_base=%s; normalize_a3r=%s; a3r_r=%s; aggregate=%s' % (normalize_base,
                                                                                             normalize_a3r,
                                                                                             a3r_r,
                                                                                             aggregate)
                    filtered_frame = filter_frame(result_total, filters)
                    if len(filtered_frame) == 0:
                        continue

                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    sns.boxplot(x="budget", y=args.scoring, hue="folder_depth_1", data=filtered_frame, ax=ax1)
                    sns.boxplot(x="budget", y=usercpu_time, hue="folder_depth_1", data=filtered_frame, ax=ax2)
                    ax2.set(yscale="log")
                    plt.title(title)
                    plt.savefig(os.path.join(output_directory_full, '%s.png' % title))


if __name__ == '__main__':
    run(parse_args())
