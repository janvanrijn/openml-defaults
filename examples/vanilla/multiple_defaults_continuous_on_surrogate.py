import argparse
import logging
import openml
import openmldefaults
import os
import pandas as pd


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/dsi/users/jv2657/experiments ~/habanero_experiments
def parse_args():
    metadata_file_svc = os.path.expanduser('~/data/openml-defaults/svc.arff')
    metadata_file_gb = os.path.expanduser('~/data/openml-defaults/gradient_boosting.arff')
    metadata_file_adaboost019 = os.path.expanduser('~/projects/openml-pimp/KDD2018/data/arff/adaboost.arff')
    metadata_file_random_forest019 = os.path.expanduser('~/projects/openml-pimp/KDD2018/data/arff/random_forest.arff')
    metadata_file_svc019 = os.path.expanduser('~/projects/openml-pimp/KDD2018/data/arff/svc.arff')
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/vanilla_defaults_vs_rs/')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--metadata_files', type=str, nargs='+', default=[metadata_file_adaboost019, metadata_file_random_forest019, metadata_file_svc019])
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--search_space_identifier', type=str, default=None)
    parser.add_argument('--minimize', action='store_true')
    parser.add_argument('--n_configurations', type=int, default=2**13)
    parser.add_argument('--normalize_base', type=str, default='MinMaxScaler')
    parser.add_argument('--normalize_a3r', type=str, default='StandardScaler')
    parser.add_argument('--a3r_r', type=int, default=1)
    parser.add_argument('--aggregate', type=str, choices=openmldefaults.experiments.AGGREGATES, default='sum')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--n_defaults', type=int, default=32)
    parser.add_argument('--n_estimators', type=int, default=64)
    parser.add_argument('--minimum_evals', type=int, default=128)
    parser.add_argument('--random_iterations', type=int, default=1)
    parser.add_argument('--run_on_surrogates', action='store_true')
    parser.add_argument('--task_limit', type=int, default=None, help='For speed')
    args_ = parser.parse_args()
    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    study = openml.study.get_study(args.study_id, 'tasks')
    if args.task_idx is None:
        all_task_ids = study.tasks
    else:
        all_task_ids = [study.tasks[args.task_idx]]

    # run random search
    for random_seed in range(args.random_iterations):
        openmldefaults.experiments.run_random_search_surrogates(
            metadata_files=args.metadata_files,
            random_seed=random_seed,
            search_space_identifier=args.search_space_identifier,
            scoring=args.scoring,
            minimize_measure=args.minimize,
            n_defaults=args.n_defaults,
            surrogate_n_estimators=args.n_estimators,
            surrogate_minimum_evals=args.minimum_evals,
            consider_runtime=False,
            output_directory=args.output_directory
        )

        for task_id in all_task_ids:
            openmldefaults.experiments.run_vanilla_surrogates_on_task(
                task_id=task_id,
                random_seed=random_seed,
                search_space_identifier=args.search_space_identifier,
                metadata_files=args.metadata_files,
                scoring=args.scoring,
                minimize_measure=args.minimize,
                n_defaults=args.n_defaults,
                n_configurations=args.n_configurations,
                aggregate=args.aggregate,
                a3r_r=args.a3r_r,
                normalize_base=args.normalize_base,
                normalize_a3r=args.normalize_a3r,
                surrogate_n_estimators=args.n_estimators,
                surrogate_minimum_evals=args.minimum_evals,
                consider_runtime=False,
                consider_a3r=False,
                run_on_surrogate=args.run_on_surrogates,
                task_limit=args.task_limit,
                output_directory=args.output_directory)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
