import argparse
import matplotlib.pyplot as plt
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
    parser.add_argument('--n_defaults_in_file', type=int, default=32)
    return parser.parse_args()


def filter_frame(df: pd.DataFrame, filters: typing.Dict):
    for filter_key, filter_value in filters.items():
        df = df.loc[df[filter_key] == filter_value]
    return df


def run(args):
    usercpu_time = 'usercpu_time_millis'
    result_total = None
    folder_constraints = {
        2: ['sum'],
        3: ['1'],
        4: ['None'],
        5: ['None']
    }
    folder_constraints = None
    for budget in [1, 2, 4, 8, 16, 32]:
        result_budget = openmldefaults.utils.results_from_folder_to_df(args.results_directory,
                                                                       args.n_defaults_in_file,
                                                                       budget,
                                                                       folder_constraints,
                                                                       False)
        print(result_budget.shape)
        result_budget['budget'] = budget
        if result_total is None:
            result_total = result_budget
        else:
            result_total = result_total.append(result_budget)

    result_total[args.scoring] = result_total[args.scoring].astype(float)
    result_total[usercpu_time] = result_total[usercpu_time].astype(float)

    os.makedirs(args.output_directory, exist_ok=True)

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

                    fig, ax = plt.subplots()
                    plt.title(title)
                    sns.boxplot(x="budget", y=args.scoring, hue="folder_depth_1", data=filtered_frame, ax=ax)
                    plt.savefig(os.path.join(args.output_directory, '%s%s.png' % (args.scoring, title)))

                    fig, ax = plt.subplots()
                    plt.title(title)
                    sns.boxplot(x="budget", y=usercpu_time, hue="folder_depth_1", data=filtered_frame, ax=ax)
                    ax.set(yscale="log")
                    plt.savefig(os.path.join(args.output_directory, '%s%s.png' % (usercpu_time, title)))


if __name__ == '__main__':
    run(parse_args())
