import argparse
import logging
import openml
import openmldefaults
import os
import pandas as pd


def parse_args():
    metadata_file = os.path.expanduser('~/data/openml-defaults/svc.arff')
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/vanilla_defaults/')
    parser.add_argument('--study_id', type=str, default='OpenML100', help='the tag to obtain the tasks from')
    parser.add_argument('--task_idx', type=int, default=None)
    parser.add_argument('--metadata_file', type=str, default=metadata_file)
    parser.add_argument('--classifier_name', type=str, default='svc', help='scikit-learn flow name')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--search_space_identifier', type=str, default='small')
    parser.add_argument('--minimize', action='store_true')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--n_defaults', type=int, default=32)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--task_limit', type=int, default=None, help='For speed')
    args_ = parser.parse_args()
    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    study = openml.study.get_study(args.study_id, 'tasks')
    for normalize_base in [None, 'MinMaxScaler', 'StandardScaler']:
        for normalize_a3r in [None, 'MinMaxScaler', 'StandardScaler']:
            for a3r_r in [1, 2, 3, 4]:
                for aggregate in ['sum', 'median']:
                    if args.task_idx is None:
                        for task_idx in range(len(study.tasks)):
                            task_id = study.tasks[task_idx]
                            openmldefaults.experiments.run_vanilla_surrogates_on_task(
                                task_id=task_id,
                                classifier_name=args.classifier_name,
                                random_seed=args.random_seed,
                                search_space_identifier=args.search_space_identifier,
                                metadata_file=args.metadata_file,
                                resized_grid_size=args.resized_grid_size,
                                scoring=args.scoring,
                                minimize=args.minimize,
                                n_defaults=args.n_defaults,
                                aggregate=aggregate,
                                a3r_r=a3r_r,
                                normalize_base=normalize_base,
                                normalize_a3r=normalize_a3r,
                                task_limit=args.task_limit,
                                output_directory=args.output_directory)
                    else:
                        task_id = study.tasks[args.task_idx]
                        openmldefaults.experiments.run_vanilla_surrogates_on_task(
                            task_id=task_id,
                            classifier_name=args.classifier_name,
                            random_seed=args.random_seed,
                            search_space_identifier=args.search_space_identifier,
                            metadata_file=args.metadata_file,
                            resized_grid_size=args.resized_grid_size,
                            scoring=args.scoring,
                            minimize=args.minimize,
                            n_defaults=args.n_defaults,
                            aggregate=aggregate,
                            a3r_r=a3r_r,
                            normalize_base=normalize_base,
                            normalize_a3r=normalize_a3r,
                            task_limit=args.task_limit,
                            output_directory=args.output_directory)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
