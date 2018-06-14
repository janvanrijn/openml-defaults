import argparse
import matplotlib.pyplot as plt
import numpy as np
import openmldefaults
import os
import pickle


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/train_svm.feather')
    parser.add_argument('--dataset_test_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/test_svm.feather')
    parser.add_argument('--flip_performances', action='store_true')
    parser.add_argument('--resized_grid_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    parser.add_argument('--c_executable', type=str, default='../c/main')
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--restricted_num_tasks', type=int, default=None)
    parser.add_argument('--num_defaults', type=int, default=3)
    parser.add_argument('--vs_strategy', type=str, default='cpp_bruteforce')
    return parser.parse_args()


def simulate_random_search(df, num_iterations, num_repeats):
    num_obs, num_tasks = df.shape
    all_results = np.zeros((num_repeats, num_tasks), dtype=float)
    for i in range(num_repeats):
        all_results[i] = openmldefaults.utils.selected_set_index(df, np.random.choice(num_obs, num_iterations)).values
    result = np.mean(all_results, axis=0)
    assert result.shape == (num_tasks,)
    return result


def plot(data, title, output_file):
    fig, ax = plt.subplots(1, len(data), figsize=(8 * len(data), 6))
    fig.text(0, 1, title, va='top', ha='left')
    for idx, evaluation_set in enumerate(data.keys()):
        if len(data) > 1:
            curr_ax = ax[idx]
        else:
            curr_ax = ax

        labels = []
        series = []

        for model in data[evaluation_set]:
            labels.append(model)
            series.append(data[evaluation_set][model])

        # basic plot
        curr_ax.boxplot(series)
        curr_ax.set_xticklabels(labels, rotation=45, ha='right')
        curr_ax.set_title(evaluation_set)
        # draws horizontal line
        curr_ax.axhline(y=0)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def run(dataset_train_path, dataset_test_path, flip_performances, params, resized_grid_size, num_defaults, output_dir, vs_strategy):
    frames = dict()
    frames['train'] = openmldefaults.utils.load_dataset(dataset_train_path, params, None, flip_performances)

    if dataset_test_path is not None:
        frames['test'] = openmldefaults.utils.load_dataset(dataset_test_path, params, None, flip_performances)

    train_data_name = os.path.basename(dataset_train_path)
    data_dir = os.path.join(output_dir, train_data_name)

    if not os.path.isdir(data_dir):
        raise ValueError('dir does not exists: %s' % data_dir)

    setup_name = openmldefaults.utils.get_setup_dirname(resized_grid_size, num_defaults)
    results = dict()
    runtime = {'train': dict()}
    for strategy in os.listdir(data_dir):
        setup_dir = os.path.join(data_dir, strategy, setup_name)
        if os.path.isdir(setup_dir):
            results_file = os.path.join(setup_dir, 'results.pkl')
            with open(results_file, 'rb') as fp:
                strategy_results = pickle.load(fp)

            for eval_frame, df in frames.items():
                if eval_frame not in results:
                    results[eval_frame] = dict()
                results[eval_frame][strategy] = openmldefaults.utils.selected_set(df, strategy_results['defaults']).values
            runtime['train'][strategy] = strategy_results['run_time']

    for eval_frame, df in frames.items():
        for budget in range(4, 0, -1):
            results[eval_frame]['random_search_budget_x%d' % budget] = simulate_random_search(df, num_defaults * budget, 100)

    title = '%s (%d defaults)' % (os.path.basename(dataset_train_path), num_defaults)
    plot(results, title, os.path.join(output_dir, 'absolute_%s_%s.png' % (train_data_name, setup_name)))
    plot(runtime, title, os.path.join(output_dir, 'runtime_%s_%s.png' % (train_data_name, setup_name)))

    results_relative = dict()
    for eval_frame, df in frames.items():
        results_relative[eval_frame] = dict()
        for strategy in results[eval_frame]:
            if strategy == vs_strategy:
                continue
            current_result = results[eval_frame][vs_strategy] - results[eval_frame][strategy]
            results_relative[eval_frame]['vs_%s' % strategy] = current_result
    plot(results_relative, title, os.path.join(output_dir, 'relative_%s_%s.png' % (train_data_name, setup_name)))


if __name__ == '__main__':
    args = parse_args()
    run(args.dataset_train_path, args.dataset_test_path, args.flip_performances, args.params, args.resized_grid_size,
        args.num_defaults, args.output_dir, args.vs_strategy)
