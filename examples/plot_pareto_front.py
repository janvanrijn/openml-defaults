import argparse
import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import openmldefaults
import os
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/svm-ongrid.arff')
    parser.add_argument('--experiment_dir', type=str,
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/')
    parser.add_argument('--flip_performances', action='store_true')
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--num_params_plot', type=int, default=3)
    parser.add_argument('--resized_grid_size', type=int, default=16)
    parser.add_argument('--num_defaults', type=int, default=3)
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    parser.add_argument('--condition_on', type=json.loads, default=None)
    return parser.parse_args()


def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def plot_numeric(ax, name, x_series_dominated, y_series_dominated):
    if len(x_series_dominated) > 0:
        ax.scatter(rand_jitter(x_series_dominated.tolist()),
                   rand_jitter(y_series_dominated.tolist()),
                   label=name)


def run(args):
    df_orig = openmldefaults.utils.load_dataset(args.dataset_path,
                                                args.params,
                                                args.resized_grid_size,
                                                args.flip_performances,
                                                args.condition_on)
    print(df_orig.shape)
    category_df = collections.OrderedDict()
    category_df['pareto'], category_df['dominated'] = openmldefaults.utils.simple_cull(df_orig, openmldefaults.utils.dominates_min)
    json_serialized = ''
    if args.condition_on is not None:
        for param, value in args.condition_on.items():
            json_serialized += '__' + param + '_' + str(value)

    plot_params = args.params[:args.num_params_plot]
    params_per_axis = len(plot_params)-1
    fig, axes = plt.subplots(params_per_axis, params_per_axis, figsize=(params_per_axis*8, params_per_axis*6))

    dataset_name = os.path.basename(args.dataset_path)
    experiment_dataset_dir = os.path.join(args.experiment_dir, dataset_name)
    for solver_name in os.listdir(experiment_dataset_dir):
        results_file = os.path.join(experiment_dataset_dir,
                                    solver_name,
                                    openmldefaults.utils.get_setup_dirname(args.resized_grid_size, args.num_defaults),
                                    'results.pkl')
        if not os.path.isfile(results_file):
            print(openmldefaults.utils.get_time(), 'Warning: No results.pkl for %s' % solver_name)
            continue
        with open(results_file, 'rb') as fp:
            results = pickle.load(fp)
            category_name = '%s_d%d' % (solver_name, args.num_defaults)
            category_df[category_name] = df_orig.loc[results['defaults']]
            if len(category_df[category_name]) != args.num_defaults:
                raise ValueError()

    for i in range(len(plot_params)-1):
        for j in range(i + 1, len(plot_params)):
            current_ax = axes[i, j-1]
            current_ax.set_xlabel(plot_params[i])
            current_ax.set_ylabel(plot_params[j])

            for name, df in category_df.items():
                current = df.reset_index()
                x_series = current[plot_params[i]]
                y_series = current[plot_params[j]]

                x_numeric = np.issubdtype(x_series.dtype, np.number)
                y_numeric = np.issubdtype(y_series.dtype, np.number)

                if x_numeric and y_numeric:
                    plot_numeric(current_ax, name, x_series, y_series)
                elif x_numeric:
                    print(y_series)
                    raise ValueError('Non numeric column: %s' % (plot_params[j]))
                elif y_numeric:
                    print(x_series)
                    raise ValueError('Non numeric column: %s' % (plot_params[i]))
                else:
                    print(x_series)
                    print(y_series)
                    raise ValueError('Non numeric columns: %s, %s' % (plot_params[i], plot_params[j]))

    axes[0, 0].legend()
    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(os.path.join(args.output_dir, 'params_%s%s.png' % (dataset_name,
                                                                   json_serialized)))


if __name__ == '__main__':
    run(parse_args())
