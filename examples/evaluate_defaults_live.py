import argparse
import collections
import copy
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
    parser.add_argument('--normalize', action="store_true", default=True)
    parser.add_argument('--input_dir', type=str, default=os.path.expanduser('~') + '/habanero_experiments/openml-defaults')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/habanero_experiments/openml-defaults')
    parser.add_argument('--vs_strategy', type=str, default='greedy')
    return parser.parse_args()


def plot(defaults_strategy_task_score, output_file):
    n_figs = len(defaults_strategy_task_score)
    fig = plt.figure(figsize=(4*n_figs, 6))
    axes = [fig.add_subplot(1, n_figs, i) for i in range(1, n_figs + 1)]
    for i, (num_defaults, strategy_task_score) in enumerate(defaults_strategy_task_score.items()):
        print([list(task_score.values()) for task_score in strategy_task_score.values()])
        axes[i].boxplot([list(task_score.values()) for task_score in strategy_task_score.values()])
        axes[i].set_xticklabels([strategy for strategy in strategy_task_score.keys()], rotation=45, ha='right')
        axes[i].set_title(str(num_defaults) + ' defaults')
    axes[0].set_ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(openmldefaults.utils.get_time(), 'saved to', output_file)


def normalize_scores(defaults_strategy_task_score, task_minscore, task_maxscore):
    result = copy.deepcopy(defaults_strategy_task_score)
    for defaults, strategy_task_score in defaults_strategy_task_score.items():
        for strategy, task_score in strategy_task_score.items():
            for task, score in task_score.items():
                mi = task_minscore[task]
                ma = task_maxscore[task]
                if mi < ma:
                    normalized_score = (score - mi) / (ma - mi)
                else:
                    normalized_score = 1.0
                result[defaults][strategy][task] = normalized_score
    return result


def run():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        raise ValueError()
    strategies_dir = os.path.join(args.input_dir, args.dataset_name, 'live_random_search')
    if not os.path.isdir(strategies_dir):
        raise ValueError()

    results = collections.OrderedDict()
    task_minscores = dict()
    task_maxscores = dict()
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
                    if task_id not in task_minscores or accuracy_avg < task_minscores[task_id]:
                        task_minscores[task_id] = accuracy_avg
                    if task_id not in task_maxscores or accuracy_avg > task_maxscores[task_id]:
                        task_maxscores[task_id] = accuracy_avg

                    if num_defaults not in results:
                        results[num_defaults] = collections.OrderedDict()
                    if strategy not in results[num_defaults]:
                        results[num_defaults][strategy] = dict()
                    results[num_defaults][strategy][task_id] = accuracy_avg
                except ValueError:
                    # experiment not terminated yet
                    pass
            print(openmldefaults.utils.get_time(),
                  'Strategy %s %d defaults, loaded %d tasks' % (strategy, num_defaults,
                                                                len(results[num_defaults][strategy])))
    if args.normalize:
        results = normalize_scores(results, task_minscores, task_maxscores)

    filename = "%s_live%s.png" % (args.dataset_name, '_normalized' if args.normalize is True else '')
    output_file = os.path.join(args.output_dir, filename)
    plot(results, output_file)


if __name__ == '__main__':
    run()
