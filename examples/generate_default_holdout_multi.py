import argparse
import openmldefaults
import os

from examples.generate_defaults import run as generate_defaults


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--scoring', type=str, default=None)
    parser.add_argument('--max_num_defaults', type=int, default=10)
    parser.add_argument('--c_executable', type=str, default='../c/main')
    parser.add_argument('--cv_iterations', type=int, default=10)
    parser.add_argument('--cv_iteration', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    parser.add_argument('--run_bb', action='store_true')
    parser.add_argument('--run_mip', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    datasets = {
        ('adaboost', 'predictive_accuracy'):
            (os.path.expanduser('~') + '/data/openml-defaults/surrogate__adaboost__predictive_accuracy__c8.arff',
             100,
             True,
             ['classifier__algorithm', 'classifier__learning_rate', 'classifier__base_estimator__max_depth',
              'classifier__n_estimators', 'imputation__strategy']),
        ('random_forest', 'predictive_accuracy'):
            (os.path.expanduser('~') + '/data/openml-defaults/surrogate__random_forest__predictive_accuracy__c8.arff',
             94,
             True,
             ['classifier__bootstrap', 'classifier__criterion', 'classifier__max_depth', 'classifier__max_features',
              'classifier__max_leaf_nodes', 'classifier__min_samples_leaf', 'classifier__min_samples_split',
              'classifier__min_weight_fraction_leaf', 'classifier__n_estimators', 'imputation__strategy']),
        ('adaboost', 'f_measure'):
            (os.path.expanduser('~') + '/data/openml-defaults/surrogate__adaboost__f_measure__c8.arff',
             100,
             True,
             ['classifier__algorithm', 'classifier__learning_rate', 'classifier__base_estimator__max_depth',
              'classifier__n_estimators', 'imputation__strategy']),
        ('random_forest', 'f_measure'):
            (os.path.expanduser('~') + '/data/openml-defaults/surrogate__random_forest__f_measure__c8.arff',
             94,
             True,
             ['classifier__bootstrap', 'classifier__criterion', 'classifier__max_depth', 'classifier__max_features',
              'classifier__max_leaf_nodes', 'classifier__min_samples_leaf', 'classifier__min_samples_split',
              'classifier__min_weight_fraction_leaf', 'classifier__n_estimators', 'imputation__strategy']),
        ('libsvm_svc', 'predictive_accuracy'):
            (os.path.expanduser('~') + '/data/openml-defaults/surrogate__libsvm_svc__predictive_accuracy__c8.arff',
             100,
             True,
             ['C', 'coef0', 'degree', 'gamma', 'kernel']),
        ('libsvm_svc', 'f_measure'):
            (os.path.expanduser('~') + '/data/openml-defaults/surrogate__libsvm_svc__f_measure__c8.arff',
             100,
             True,
             ['classifier__C', 'classifier__coef0', 'classifier__degree', 'classifier__gamma', 'classifier__kernel']),

    }

    datasets_to_run = datasets
    if args.dataset is not None and args.scoring is not None:
        key = (args.dataset, args.scoring)
        datasets_to_run = {key: datasets[key]}
    elif args.dataset is not None:
        key = args.dataset
        datasets_to_run = {key: datasets[key]}
        # TODO: won't work when just dataset is given
    elif args.scoring is not None:
        raise ValueError('Not supported')

    models = [openmldefaults.models.GreedyDefaults()]
    determine_pareto = False
    if args.run_bb:
        models.append(openmldefaults.models.CppDefaults(args.c_executable, True))
        determine_pareto = True
    if args.run_mip:
        models.append(openmldefaults.models.MipDefaults('GLPK_CMD'))
        determine_pareto = True

    for num_defaults in range(1, args.max_num_defaults + 1):
        for (dataset_train_path, num_tasks, flip_performances, params) in datasets_to_run.values():

            cv_iterations = list(range(args.cv_iterations))
            if args.cv_iteration is not None:
                cv_iterations = [args.cv_iteration]

            for cv_iteration in cv_iterations:
                holdout_tasks = openmldefaults.utils.get_cv_indices(num_tasks, args.cv_iterations, cv_iteration)
                generate_defaults(dataset_train_path, flip_performances, params, args.resized_grid_size, num_defaults,
                                  models, args.output_dir, holdout_tasks, determine_pareto=determine_pareto)
