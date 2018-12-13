import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import openmldefaults
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--results_directory', type=str, default=os.path.expanduser('~/habanero_experiments/openml-defaults'))
    parser.add_argument('--output_directory', type=str, default=os.path.expanduser('~/experiments/openml-defaults'))
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--n_defaults_in_file', type=int, default=32)
    return parser.parse_args()


def run(args):
    usercpu_time = 'usercpu_time_millis'
    result_total = None
    for budget in [1, 2, 4, 8, 16, 32]:
        result_budget = openmldefaults.utils.results_from_folder_to_df(args.results_directory,
                                                                       args.n_defaults_in_file,
                                                                       budget)
        result_budget['budget'] = budget
        if result_total is None:
            result_total = result_budget
        else:
            result_total = result_total.append(result_budget)

    result_total[args.scoring] = result_total[args.scoring].astype(float)
    result_total[usercpu_time] = result_total[usercpu_time].astype(float)

    os.makedirs(args.output_directory, exist_ok=True)

    fig, ax = plt.subplots()
    sns.boxplot(x="budget", y=args.scoring, hue="folder_depth_1", data=result_total, ax=ax)
    plt.savefig(os.path.join(args.output_directory, '%s.png' % args.scoring))

    fig, ax = plt.subplots()
    sns.boxplot(x="budget", y=usercpu_time, hue="folder_depth_1", data=result_total, ax=ax)
    ax.set(yscale="log")
    plt.savefig(os.path.join(args.output_directory, '%s.png' % usercpu_time))


if __name__ == '__main__':
    run(parse_args())
