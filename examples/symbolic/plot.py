import argparse
import os
import pandas as pd
import pickle
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults/symbolic_defaults/')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/Desktop/')
    parser.add_argument('--per_task', action='store_true', default=True)
    return parser.parse_args()


def get_results_train(directory):
    with open(os.path.join(directory, 'results_all.pkl'), 'rb') as fp:
        results = pickle.load(fp)

    baseline_result = results['baseline_results_per_task']
    best_avg_score = results['baseline_avg_performance']
    best_result = None

    for result in results['symbolic_defaults']:
        if result['avg_performance'] > best_avg_score:
            best_avg_score = result['avg_performance']
            best_result = result['results_per_task']

    data = []
    for idx in range(len(best_result)):
        data.append({'defaults_type': 'symbolic', 'task_idx': idx, 'evaluation': best_result[idx]})
        data.append({'defaults_type': 'vanilla', 'task_idx': idx, 'evaluation': baseline_result[idx]})

    df = pd.DataFrame(data=data)
    return df


def get_results_holdout(directory):
    data = []
    for file in os.listdir(directory):
        if '_all' in file:
            continue
        with open(os.path.join(directory, file), 'rb') as fp:
            results = pickle.load(fp)
        task_id = file.split('.')[0].split('_')[1]

        baseline_holdout = results['baseline_holdout_score']
        best_avg_score = results['baseline_avg_performance']
        best_holdout = None

        for result in results['symbolic_defaults']:
            if result['avg_performance'] > best_avg_score:
                best_avg_score = result['avg_performance']
                best_holdout = result['holdout_score']

        data.append({'defaults_type': 'symbolic', 'task_idx': task_id, 'evaluation': best_holdout})
        data.append({'defaults_type': 'vanilla', 'task_idx': task_id, 'evaluation': baseline_holdout})

    df = pd.DataFrame(data=data)
    return df


def run(args):
    if args.per_task:
        df = get_results_holdout(args.results_dir)
    else:
        df = get_results_train(args.results_dir)

    sns_plot = sns.boxplot(x='defaults_type', y='evaluation', data=df)
    fig = sns_plot.get_figure()
    fig.savefig(os.path.join(args.output_dir, 'results.pdf'))

    df = df.pivot(index='task_idx', columns='defaults_type', values='evaluation')
    df['difference'] = df.apply(lambda x: x['symbolic'] - x['vanilla'], axis=1)
    wins = 0
    draws = 0
    loses = 0

    for diff in df['difference']:
        if diff > 0:
            wins += 1
        elif diff < 0:
            loses += 1
        else:
            draws += 1
    print('%d wins; %d draws; %d loses' % (wins, draws, loses))


if __name__ == '__main__':
    run(parse_args())
