import argparse
import openmldefaults
import os

from examples.generate_defaults import run as generate_defaults


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resized_grid_size', type=int, default=16)
    parser.add_argument('--max_num_defaults', type=int, default=10)
    parser.add_argument('--c_executable', type=str, default='../c/main')
    parser.add_argument('--cv_iterations', type=int, default=10)
    parser.add_argument('--cv_iteration', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    datasets = [
        (os.path.expanduser('~') + '/data/openml-defaults/svm-ongrid.arff',
         50,
         True,
         ['kernel_rbf', 'kernel_poly', 'kernel_linear', 'c', 'gamma', 'degree'])
    ]

    models = [
        openmldefaults.models.CppDefaults(args.c_executable, True),
        openmldefaults.models.GreedyDefaults()
    ]

    for num_defaults in range(1, args.max_num_defaults + 1):
        for (dataset_train_path, num_tasks, flip_performances, params) in datasets:

            cv_iterations = list(range(args.cv_iterations))
            if args.cv_iteration is not None:
                cv_iterations = [args.cv_iteration]

            for cv_iteration in cv_iterations:
                holdout_tasks = openmldefaults.utils.get_cv_indices(num_tasks, args.cv_iterations, cv_iteration)
                generate_defaults(dataset_train_path, flip_performances, params, args.resized_grid_size, num_defaults,
                                  models, args.output_dir, holdout_tasks)
