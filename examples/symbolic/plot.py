import argparse
import os
import pandas as pd
import pickle
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=os.path.expanduser('~') + '/habanero_experiments/openml-defaults/symbolic_defaults')
    parser.add_argument('--classifier_name', type=str, default='libsvm_svc')

    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/Desktop/')
    parser.add_argument('--per_task', action='store_true', default=True)
    return parser.parse_args()


def formula_str(result):
    if result['transform_fn'] == 'inverse_transform_fn':
        return '%f / %s ' % (result['transform_alpha_value'], result['transform_meta_feature'])
    elif result['transform_fn'] == 'power_transform_fn':
        return '%f^{%s} ' % (result['transform_alpha_value'], result['transform_meta_feature'])
    elif result['transform_fn'] == 'multiply_transform_fn':
        return '%f \\cdot %s ' % (result['transform_alpha_value'], result['transform_meta_feature'])
    elif result['transform_fn'] == 'log_transform_fn':
        return '%f \\cdot \\log %s ' % (result['transform_alpha_value'], result['transform_meta_feature'])
    elif result['transform_fn'] == 'root_transform_fn':
        return '%f \\cdot \\sqrt{%s} ' % (result['transform_alpha_value'], result['transform_meta_feature'])
    else:
        raise ValueError()


def get_results_train(directory):
    with open(os.path.join(directory, 'results_all.pkl'), 'rb') as fp:
        results = pickle.load(fp)

    baseline_result = results['baseline_results_per_task']
    best_avg_score = results['baseline_avg_performance']
    best_result = None
    best_meta_formula = None

    for result in results['symbolic_defaults']:
        if result['avg_performance'] > best_avg_score:
            best_avg_score = result['avg_performance']
            best_result = result['results_per_task']
            best_meta_formula = formula_str(result)

    data = []
    for idx in range(len(best_result)):
        data.append({'defaults_type': 'symbolic', 'task_idx': idx, 'evaluation': best_result[idx], 'formula': best_meta_formula})
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
        best_meta_formula = None

        for result in results['symbolic_defaults']:
            if result['avg_performance'] > best_avg_score:
                best_avg_score = result['avg_performance']
                best_holdout = result['holdout_score']
                best_meta_formula = formula_str(result)

        data.append({'defaults_type': 'symbolic', 'task_idx': task_id, 'evaluation': best_holdout, 'formula': best_meta_formula})
        data.append({'defaults_type': 'vanilla', 'task_idx': task_id, 'evaluation': baseline_holdout})

    df = pd.DataFrame(data=data)
    return df


def run(args):
    results_dir = os.path.join(args.results_dir, args.classifier_name)
    if args.per_task:
        df_orig = get_results_holdout(results_dir)
        title = 'Results on holdout tasks'
    else:
        df_orig = get_results_train(results_dir)
        title = 'Results on train set'
    df_orig = df_orig.set_index('task_idx')

    sns_plot = sns.boxplot(x='defaults_type', y='evaluation', data=df_orig).set_title(title)
    fig = sns_plot.get_figure()
    fig.savefig(os.path.join(args.output_dir, '%s.pdf' % args.classifier_name))

    df_pivot = df_orig.pivot(index=None, columns='defaults_type', values='evaluation')
    df_pivot['difference'] = df_pivot.apply(lambda x: x['symbolic'] - x['vanilla'], axis=1)
    # join with formula
    df_pivot = df_pivot.join(df_orig[df_orig['defaults_type'] == 'symbolic']['formula'])
    df_pivot.to_csv(os.path.join(args.output_dir, '%s.csv' % args.classifier_name))
    print(df_pivot)
    wins = 0
    draws = 0
    loses = 0

    for diff in df_pivot['difference']:
        if diff > 0:
            wins += 1
        elif diff < 0:
            loses += 1
        else:
            draws += 1
    print('%d wins; %d draws; %d loses' % (wins, draws, loses))


if __name__ == '__main__':
    run(parse_args())
