import argparse
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import openmldefaults
import os
import pandas as pd
import typing


def parse_args():
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--results_directory', type=str, default=os.path.expanduser('~/habanero_experiments/openml-defaults/vanilla_vs_a3r'))
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~/experiments/openml-defaults/vanilla_vs_a3r'))
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--classifier_name', type=str, default='svc')
    parser.add_argument('--n_defaults_in_file', type=int, default=32)
    return parser.parse_args()


# for sanity checking
EXPECTED_DATASETS = 99
EXPECTED_STRATEGIES = 3
ALL_BUDGETS = [1, 2, 4, 8, 16, 32]
STRICT_CHECK = False


def filter_frame(df: pd.DataFrame, filters: typing.Dict):
    for filter_key, filter_value in filters.items():
        df = df.loc[df[filter_key] == filter_value]
    return df


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
                raise ValueError('Dataframe shapes do not collide. %s vs %s' % (df_a.shape, df_b.shape))
            difference_frame = df_a - df_b
            difference_frame['strategies'] = '%s-%s' % (difference_vals[i], difference_vals[j])
            if result is None:
                result = difference_frame
            else:
                result = result.append(difference_frame)
    result = result.reset_index()
    return result


# def make_time_curves(df, indices, x_axis_column, y_axis_column):
#     df = df.pivot_table(
#         values=y_axis_column,
#         index=indices,
#         columns=x_axis_column,
#         aggfunc=np.mean)
#     previous_column = None
#
#     # check whether columns are correctly ordered
#     for idx, column in enumerate(df.columns.values):
#         column = float(column)
#         if previous_column is not None:
#             if column <= previous_column:
#                 raise ValueError('Column idx %d value %f, previous %f' % (idx, column, previous_column))
#         previous_column = column
#     # check first column no NA's
#     assert df[df.columns[0]].isna().sum() == 0, 'First column should not contain any NA\'s'
#     df = df.fillna(method='ffill', axis=1)
#     assert sum(df.isna().sum()) == 0, 'Got %d NA\'s' % sum(df.isna().sum())
#     return df


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    usercpu_time = 'usercpu_time_millis'
    result_total = None
    folder_constraints = {
        2: ['42'],
        3: ['sum'],
        4: ['1'],
        5: ['None'],
        6: ['None']
    }
    folder_legend = {
        'folder_depth_0': 'task_id',
        'folder_depth_1': 'strategy',
        'folder_depth_2': 'random_seed',
        'folder_depth_3': 'param_aggregate',
        'folder_depth_4': 'param_a3r_r',
        'folder_depth_5': 'param_normalize_base',
        'folder_depth_6': 'param_normalize_a3r',
    }
    # folder_constraints = None
    results_directory = os.path.join(args.results_directory, args.classifier_name)
    result_curves = None
    for budget in ALL_BUDGETS:
        result_budget, result_curves = openmldefaults.utils.results_from_folder_to_df(results_directory,
                                                                                      args.n_defaults_in_file,
                                                                                      budget,
                                                                                      folder_constraints,
                                                                                      False, False)
        result_budget['budget'] = budget
        expectation = EXPECTED_DATASETS * EXPECTED_STRATEGIES
        if result_budget.shape[0] < expectation:
            msg = 'Not enough results! Expected at least %d, got %d' % (expectation,
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
    result_curves = result_curves.rename(index=str, columns=folder_legend)

    result_total[args.scoring] = result_total[args.scoring].astype(float)
    result_total[usercpu_time] = result_total[usercpu_time].astype(float)
    result_curves[args.scoring] = result_curves[args.scoring].astype(float)
    result_curves[usercpu_time] = result_curves[usercpu_time].astype(float)

    # sanity check results
    index_columns = ['task_id', 'strategy', 'random_seed', 'param_aggregate',
                     'param_a3r_r', 'param_normalize_base', 'param_normalize_a3r']
    openmldefaults.utils.check_budget_curves(result_total, index_columns, args.scoring, 'budget', 0.0, 1.0)
    openmldefaults.utils.check_budget_curves(result_total, index_columns, usercpu_time, 'budget', 1.0, None)

    output_directory_full = os.path.join(args.output_directory, args.classifier_name)
    os.makedirs(output_directory_full, exist_ok=True)

    for normalize_base in result_total['param_normalize_base'].unique():
        for normalize_a3r in result_total['param_normalize_a3r'].unique():
            for a3r_r in result_total['param_a3r_r'].unique():
                for aggregate in result_total['param_aggregate'].unique():
                    filters = {
                        'param_normalize_base': normalize_base,
                        'param_normalize_a3r': normalize_a3r,
                        'param_a3r_r': a3r_r,
                        'param_aggregate': aggregate,
                    }
                    title = 'normalize_base=%s; normalize_a3r=%s; a3r_r=%s; aggregate=%s' % (normalize_base,
                                                                                             normalize_a3r,
                                                                                             a3r_r,
                                                                                             aggregate)
                    filtered_results = filter_frame(result_total, filters)
                    filtered_curves = filter_frame(result_curves, filters)
                    strategies = filtered_results['strategy'].unique()

                    expected_rows_results = EXPECTED_DATASETS * EXPECTED_STRATEGIES * len(ALL_BUDGETS)
                    expected_rows_curves = EXPECTED_DATASETS * EXPECTED_STRATEGIES * (ALL_BUDGETS[-1] + 1)
                    if filtered_results.shape[0] != expected_rows_results:
                        msg = 'Unexpected results df shape. Expected %d rows, got %d rows' % (expected_rows_results,
                                                                                              filtered_results.shape[0])
                        logging.warning(msg)
                        if STRICT_CHECK:
                            continue
                    if filtered_curves.shape[0] != expected_rows_curves:
                        msg = 'Unexpected curve df shape. Expected %d rows, got %d rows' % (expected_rows_curves,
                                                                                            filtered_curves.shape[0])
                        logging.warning(msg)
                        if STRICT_CHECK:
                            continue

                    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(24, 12))

                    # absolute plots
                    sns.boxplot(x="budget", y=args.scoring, hue="strategy", data=filtered_results, ax=ax1)
                    sns.boxplot(x="budget", y=usercpu_time, hue="strategy", data=filtered_results, ax=ax2)
                    ax2.set(yscale="log")
                    sns.boxplot(x="budget", y='n_defaults', hue="strategy", data=filtered_results, ax=ax3)

                    # difference plots
                    results_difference = make_difference_df(filtered_results,
                                                            ['task_id', 'random_seed', 'param_aggregate', 'param_a3r_r',
                                                             'param_normalize_base', 'param_normalize_a3r', 'budget'],
                                                            'strategy')

                    sns.boxplot(x="budget", y=args.scoring, hue="strategies", data=results_difference, ax=ax4)
                    sns.boxplot(x="budget", y=usercpu_time, hue="strategies", data=results_difference, ax=ax5)
                    ax5.set(yscale="log")
                    sns.boxplot(x="budget", y='n_defaults', hue="strategies", data=results_difference, ax=ax6)

                    plt.title(title)
                    output_file = os.path.join(output_directory_full, '%s_agg.png' % title)
                    plt.savefig(output_file)
                    logging.info('stored figure to %s' % output_file)

                    # plot individual loss curves
                    rows = 10
                    cols = 10
                    fig, axes = plt.subplots(rows, cols, figsize=(24, 12))
                    task_ids = result_curves['task_id'].unique()
                    win_counts = list()
                    for idx, task_id in enumerate(task_ids):
                        current_task = result_curves.loc[result_curves['task_id'] == task_id]
                        if len(current_task) == 0:
                            # in case no results on a task
                            continue
                        legend = 'brief' if idx == 0 else None
                        sns.lineplot(x=usercpu_time, y=args.scoring, markers=True,
                                     hue="strategy", data=current_task,
                                     ax=axes[idx % cols, int(idx / cols)], legend=legend)

                        for strategy in strategies:
                            time_threshold = current_task[current_task['strategy'] == strategy][usercpu_time].max()
                            current_task_threshold = current_task.loc[current_task[usercpu_time] <= time_threshold]
                            result = current_task_threshold.groupby('strategy')['predictive_accuracy'].max()
                            for idx_a, strategy_a in enumerate(strategies):
                                for idx_b, strategy_b in enumerate(strategies):
                                    if idx_a >= idx_b:
                                        continue
                                    current = {
                                        'strategy_a': strategy_a,
                                        'strategy_b': strategy_b,
                                        'task_id': task_id,
                                        'threshold_base': strategy,
                                        'a_better': result[strategy_a] > result[strategy_b]
                                    }
                                    win_counts.append(current)
                    win_counts = pd.DataFrame(win_counts)
                    print(win_counts.groupby(['threshold_base', 'strategy_a', 'strategy_b'])['a_better'].sum())

                    output_file = os.path.join(output_directory_full, '%s_loss.png' % title)
                    plt.savefig(output_file, dpi=300)
                    logging.info('stored figure to %s' % output_file)


if __name__ == '__main__':
    run(parse_args())
