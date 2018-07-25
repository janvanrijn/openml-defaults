import argparse
import collections
import matplotlib.pyplot as plt
import openml
import openmldefaults
import os
import sklearn


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='surrogate_adaboost_c8.arff')
    parser.add_argument('--flip_performances', action='store_true', default=True)
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--max_num_defaults', type=int, default=None)
    parser.add_argument('--cv_iterations', type=int, default=10)
    parser.add_argument('--input_dir', type=str, default=os.path.expanduser('~') + '/habanero_experiments/openml-defaults')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/habanero_experiments/openml-defaults')

    parser.add_argument('--vs_strategy', type=str, default='greedy')
    return parser.parse_args()


def plot(defaults_strategy_scores, output_dir, dataset_name):
    n_figs = len(defaults_strategy_scores)
    fig = plt.figure(figsize=(8, 3*n_figs))
    axes = [fig.add_subplot(1, n_figs, i) for i in range(1, n_figs + 1)]
    for i, (num_defaults, strategy_scores) in enumerate(defaults_strategy_scores.items()):
        axes[i].boxplot([scores for scores in strategy_scores.values()])
        axes[i].set_xticklabels([strategy for strategy in strategy_scores.keys()], rotation=45, ha='right')
        axes[i].set_title(str(num_defaults) + ' defaults')
    axes[0].set_ylabel('Accuracy')
    output_file = os.path.join(output_dir, dataset_name + "_live.png")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def run():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        raise ValueError()
    strategies_dir = os.path.join(args.input_dir, args.dataset_name, 'live_random_search')
    if not os.path.isdir(strategies_dir):
        raise ValueError()

    results = collections.OrderedDict()
    for strategy in sorted(os.listdir(strategies_dir)):
        # sorted ensures sorted plots in Python 3.X, not in Python 2.7
        for setup in sorted(os.listdir(os.path.join(strategies_dir, strategy))):
            # filter out wrong setups
            grid_size_str, num_defaults_str = setup.split("_")
            if grid_size_str != "c" + str(args.resized_grid_size):
                continue
            num_defaults = int(num_defaults_str[1:])
            if args.max_num_defaults is not None and num_defaults > args.max_num_defaults:
                continue
            for task_id in os.listdir(os.path.join(strategies_dir, strategy, setup)):
                result_dir = os.path.join(strategies_dir, strategy, setup, task_id)
                try:
                    run = openml.runs.OpenMLRun.from_filesystem(result_dir, expect_model=False)
                    accuracy_scores = run.get_metric_fn(sklearn.metrics.accuracy_score)
                    accuracy_avg = sum(accuracy_scores) / len(accuracy_scores)
                    if num_defaults not in results:
                        results[num_defaults] = collections.OrderedDict()
                    if strategy not in results[num_defaults]:
                        results[num_defaults][strategy] = list()
                    results[num_defaults][strategy].append(accuracy_avg)
                except ValueError:
                    # experiment not terminated yet
                    pass
            print(openmldefaults.utils.get_time(),
                  'Strategy %s %d defaults, loaded %d tasks' % (strategy, num_defaults,
                                                                len(results[num_defaults][strategy])))
    plot(results, args.output_dir, args.dataset_name)


if __name__ == '__main__':
    run()
