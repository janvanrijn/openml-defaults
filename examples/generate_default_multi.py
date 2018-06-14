import argparse
import os

from examples.generate_defaults import run as generate_defaults
from examples.evaluate_defaults import run as evaluate_defaults


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resized_grid_size', type=int, default=16)
    parser.add_argument('--max_num_defaults', type=int, default=10)
    parser.add_argument('--c_executable', type=str, default='../c/main')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    parser.add_argument('--vs_strategy', type=str, default='cpp_bruteforce')
    return parser.parse_args()


if __name__ == '__main__':
    datasets = [
        (os.path.expanduser('~') + '/data/openml-defaults/train_svm.feather',
         os.path.expanduser('~') + '/data/openml-defaults/test_svm.feather',
         False,
         ['kernel', 'cost', 'gamma', 'degree']),
        (os.path.expanduser('~') + '/data/openml-defaults/svm-ongrid.arff',
         None,
         True,
         ['kernel_rbf', 'kernel_poly', 'kernel_linear', 'c', 'gamma', 'degree'])
    ]
    args = parse_args()
    for num_defaults in range(1, args.max_num_defaults + 1):
        for (dataset_train_path, dataset_test_path, flip_performances, params) in datasets:
            generate_defaults(dataset_train_path, flip_performances, params, args.resized_grid_size, num_defaults,
                              args.c_executable, args.output_dir)
            evaluate_defaults(dataset_train_path, dataset_test_path, flip_performances, params, args.resized_grid_size,
                              num_defaults, args.output_dir, args.vs_strategy)
