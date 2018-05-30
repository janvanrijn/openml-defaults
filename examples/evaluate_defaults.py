import argparse
import feather
import matplotlib.pyplot as plt
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
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    parser.add_argument('--c_executable', type=str, default='../c/main')
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--resized_grid_size', type=int, default=16)
    parser.add_argument('--restricted_num_tasks', type=int, default=None)
    parser.add_argument('--num_defaults', type=int, default=7)
    return parser.parse_args()


def plot(data, output_file):
    fig, ax = plt.subplots(1, len(data), figsize=(8 * len(data), 6))
    for idx, evaluation_set in enumerate(data.keys()):
        labels = []
        series = []

        for model in data[evaluation_set]:
            labels.append(model)
            series.append(data[evaluation_set][model])

        # basic plot
        ax[idx].boxplot(series)
        ax[idx].set_xticklabels(labels, rotation=45, ha='right')
        ax[idx].set_title(evaluation_set)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def run(args):

    df_train = feather.read_dataframe(args.dataset_test_path)
    df_train = df_train.set_index(args.params)
    df_test = feather.read_dataframe(args.dataset_test_path)
    df_test = df_test.set_index(args.params)
    frames = {'train': df_train, 'test': df_test}

    train_data_name = os.path.basename(args.dataset_train_path)
    data_dir = os.path.join(args.output_dir, train_data_name)

    if not os.path.isdir(data_dir):
        raise ValueError()

    setup_name = openmldefaults.utils.get_setup_dirname(args)
    results = {'train': dict(), 'test': dict()}
    for strategy in os.listdir(data_dir):
        setup_dir = os.path.join(data_dir, strategy, setup_name)
        if os.path.isdir(setup_dir):
            results_file = os.path.join(setup_dir, 'results.pkl')
            results['test'][strategy] = dict()
            with open(results_file, 'rb') as fp:
                strategy_results = pickle.load(fp)
            print(results_file, strategy_results['objective'])

            for name, df in frames.items():
                results[name][strategy] = openmldefaults.utils.selected_set(df, strategy_results['defaults']).values
    plot(results, os.path.join(args.output_dir, '%s_%s.png' % (train_data_name, setup_name)))
    print(results)


if __name__ == '__main__':
    run(parse_args())
