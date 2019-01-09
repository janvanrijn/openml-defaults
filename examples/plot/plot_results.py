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


# for sanity checking
EXPECTED_DATASETS = 99
EXPECTED_STRATEGIES = 3
ALL_BUDGETS = [1, 2, 4, 8, 16, 32]


def filter_frame(df: pd.DataFrame, filters: typing.Dict):
    for filter_key, filter_value in filters.items():
        df = df.loc[df[filter_key] == filter_value]
    return df


def check_budget_curves(df: pd.DataFrame, values_column: str):
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


def make_difference_df(df: pd.DataFrame, keys: typing.List, difference_field: str):
    result = None
    difference_vals = df[difference_field].unique()
    df = df.set_index(keys)

    for i in range(len(difference_vals)-1):
        for j in range(i+1, len(difference_vals)):
            df_a = df.loc[df[difference_field] == difference_vals[i]]
            df_b = df.loc[df[difference_field] == difference_vals[j]]
            del df_a[difference_field]
            del df_b[difference_field]
            if df_a.shape != df_b.shape:
                raise ValueError()
            difference_frame = df_a - df_b
            difference_frame['strategies'] = '%s-%s' % (difference_vals[i], difference_vals[j])
            if result is None:
                result = difference_frame
            else:
                result = result.append(difference_frame)
    result = result.reset_index()
    return result


def make_time_curves(df, indices, x_axis_column, y_axis_column):
    df = df.pivot_table(
        values=y_axis_column,
        index=indices,
        columns=x_axis_column,
        aggfunc=np.mean)
    previous_column = None

    # check whether columns are correctly ordered
    for idx, column in enumerate(df.columns.values):
        column = float(column)
        if previous_column is not None:
            if column <= previous_column:
                raise ValueError('Column idx %d value %f, previous %f' % (idx, column, previous_column))
        previous_column = column
    # check first column no NA's
    assert df[df.columns[0]].isna().sum() == 0, 'First column should not contain any NA\'s'
    df = df.fillna(method='ffill', axis=1)
    assert sum(df.isna().sum()) == 0, 'Got %d NA\'s' % sum(df.isna().sum())
    return df


def run(args):
    usercpu_time = 'usercpu_time_millis'
    result_total = None
    folder_constraints = {
        2: ['sum'],
        3: ['2'],
        4: ['None'],
        5: ['None']
    }
    # folder_constraints = None
    results_directory = os.path.join(args.results_directory, args.classifier_name)
    result_curves = None
    for budget in ALL_BUDGETS:
        result_budget, result_curves = openmldefaults.utils.results_from_folder_to_df(results_directory,
                                                                                      args.n_defaults_in_file,
                                                                                      budget,
                                                                                      folder_constraints,
                                                                                      False)
        result_budget['budget'] = budget
        if result_budget.shape[0] < EXPECTED_DATASETS * EXPECTED_STRATEGIES:
            raise ValueError('Not enough results! Expected at least %d, got %d' % (EXPECTED_DATASETS * EXPECTED_STRATEGIES,
                                                                                   result_budget.shape[0]))
        if result_total is None:
            result_total = result_budget
        else:
            result_total = result_total.append(result_budget)

    result_total[args.scoring] = result_total[args.scoring].astype(float)
    result_total[usercpu_time] = result_total[usercpu_time].astype(float)
    result_curves[args.scoring] = result_curves[args.scoring].astype(float)
    result_curves[usercpu_time] = result_curves[usercpu_time].astype(float)

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
                    filtered_results = filter_frame(result_total, filters)
                    filtered_curves = filter_frame(result_curves, filters)

                    expected_rows_results = EXPECTED_DATASETS * EXPECTED_STRATEGIES * len(ALL_BUDGETS)
                    expected_rows_curves = EXPECTED_DATASETS * EXPECTED_STRATEGIES * (ALL_BUDGETS[-1] + 1)
                    if filtered_results.shape[0] != expected_rows_results:
                        logging.warning('Unexpected results df shape. Expected %d rows, got %d rows' % (expected_rows_results,
                                                                                                        filtered_results.shape[0]))
                        continue
                    if filtered_curves.shape[0] != expected_rows_curves:
                        logging.warning('Unexpected curve df shape. Expected %d rows, got %d rows' % (expected_rows_curves,
                                                                                                      filtered_curves.shape[0]))
                        continue
                    time_curves = make_time_curves(filtered_curves, ['folder_depth_0', 'folder_depth_1'], usercpu_time, args.scoring)
                    time_curves = time_curves.reset_index()
                    del time_curves['folder_depth_0']
                    print(time_curves.groupby('folder_depth_1').mean().T)

                    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, _, _)) = plt.subplots(3, 3, figsize=(24, 18))

                    # absolute plots
                    sns.boxplot(x="budget", y=args.scoring, hue="folder_depth_1", data=filtered_results, ax=ax1)
                    sns.boxplot(x="budget", y=usercpu_time, hue="folder_depth_1", data=filtered_results, ax=ax2)
                    ax2.set(yscale="log")
                    sns.boxplot(x="budget", y='n_defaults', hue="folder_depth_1", data=filtered_results, ax=ax3)

                    # difference plots
                    results_difference = make_difference_df(filtered_results,
                                                            ['folder_depth_0', 'folder_depth_2', 'folder_depth_3',
                                                             'folder_depth_4', 'folder_depth_5', 'budget'],
                                                            'folder_depth_1')

                    sns.boxplot(x="budget", y=args.scoring, hue="strategies", data=results_difference, ax=ax4)
                    sns.boxplot(x="budget", y=usercpu_time, hue="strategies", data=results_difference, ax=ax5)
                    ax5.set(yscale="log")
                    sns.boxplot(x="budget", y='n_defaults', hue="strategies", data=results_difference, ax=ax6)

                    sns.lineplot(x=usercpu_time, y=args.scoring,
                                 hue="folder_depth_1", data=filtered_curves, ax=ax7)

                    plt.title(title)
                    plt.savefig(os.path.join(output_directory_full, '%s.png' % title))


if __name__ == '__main__':
    run(parse_args())
