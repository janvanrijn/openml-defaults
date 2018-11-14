import argparse
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import openmldefaults
import os
import pandas as pd


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/surrogate__adaboost__predictive_accuracy__c8.arff')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--input_file', type=str, default=os.path.expanduser('~') + '/habanero_experiments/openml-defaults/20180826/surrogate__adaboost__predictive_accuracy__c8.arff/live_random_search/results.csv')
    return parser.parse_args()


def plot(df, y_label, output_file):
    sns_plot = sns.boxplot(x='n_defaults', y='evaluation', hue='strategy_type', data=df, palette="Set3")
    fig = sns_plot.get_figure()
    fig.savefig(output_file)
    plt.clf()
    print(openmldefaults.utils.get_time(), 'saved to', output_file)


def count_results(df):
    print(df.groupby(["strategy_type", "n_defaults"]).agg("count"))


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

    df['strategy_type'] = df['strategy_name'].apply(lambda x: x.split('__')[0])
    df['n_defaults'] = df['strategy_name'].apply(lambda x: int(x.split('__')[1]))
    df = df.groupby(['strategy_name', 'task_id', 'strategy_type', 'n_defaults']).mean().reset_index()
    # removed unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    df = df.loc[df['configuration_specification'] == args.resized_grid_size]

    # print statistics
    count_results(df)

    # normalize
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
