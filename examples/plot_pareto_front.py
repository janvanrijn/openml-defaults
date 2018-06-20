import argparse
import matplotlib.pyplot as plt
import numpy as np
import openmldefaults
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/svm-ongrid.arff')
    parser.add_argument('--flip_performances', action='store_true')
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--resized_grid_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    return parser.parse_args()


def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def plot_numeric(ax, x_series_dominated, y_series_dominated, x_series_pareto, y_series_pareto):
    ax.scatter(rand_jitter(x_series_dominated.tolist()), rand_jitter(y_series_dominated.tolist()), color='green')
    ax.scatter(rand_jitter(x_series_pareto.tolist()), rand_jitter(y_series_pareto.tolist()), color='red')


def run(args):
    df = openmldefaults.utils.load_dataset(args.dataset_path,
                                           args.params,
                                           args.resized_grid_size,
                                           args.flip_performances)

    df, dominated = openmldefaults.utils.simple_cull(df, openmldefaults.utils.dominates_min)

    df = df.reset_index()
    dominated = dominated.reset_index()

    plot_params = ['c', 'gamma', 'degree']
    fig, axes = plt.subplots(len(plot_params)-1, len(plot_params)-1)

    for i in range(len(plot_params)-1):
        for j in range(i + 1, len(plot_params)):
            x_series_dominated = dominated[plot_params[i]]
            y_series_dominated = dominated[plot_params[j]]
            x_series_pareto = df[plot_params[i]]
            y_series_pareto = df[plot_params[j]]

            if x_series_dominated.dtype != x_series_pareto.dtype:
                raise ValueError()
            if y_series_dominated.dtype != y_series_pareto.dtype:
                raise ValueError()

            current_ax = axes[i, j-1]
            plot_numeric(current_ax, x_series_dominated, y_series_dominated, x_series_pareto, y_series_pareto)
            current_ax.set_xlabel(plot_params[i])
            current_ax.set_ylabel(plot_params[j])
    plt.show()


if __name__ == '__main__':
    run(parse_args())
