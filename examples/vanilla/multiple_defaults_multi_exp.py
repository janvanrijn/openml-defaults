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
    parser.add_argument('--metadata_files', type=str, nargs='+', default=metadata_file)
    parser.add_argument('--classifier_name', type=str, default='svc', help='scikit-learn flow name')
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--search_space_identifier', type=str, default='small')
    parser.add_argument('--minimize', action='store_true')
    parser.add_argument('--resized_grid_size', type=int, default=8)
    parser.add_argument('--n_defaults', type=int, default=32)
    parser.add_argument('--n_configurations', type=int, default=2**13)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--task_limit', type=int, default=None, help='For speed')
    parser.add_argument('--n_estimators', type=int, default=64)
    parser.add_argument('--minimum_evals', type=int, default=128)
    args_ = parser.parse_args()
    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    normalize_bases = [None, 'MinMaxScaler', 'StandardScaler']
    normalize_a3rs = [None, 'MinMaxScaler', 'StandardScaler']
    a3r_rs = [1, 2, 3, 4]
    aggregates = ['sum', 'median']
    total_experiments = len(normalize_bases) * len(normalize_a3rs) * len(a3r_rs) * len(aggregates)

    study = openml.study.get_study(args.study_id, 'tasks')
    if args.task_idx is None:
        all_task_ids = study.tasks
    else:
        all_task_ids = [study.tasks[args.task_idx]]

    exp_index = 0  # just for logging
    for normalize_base in normalize_bases:
        for normalize_a3r in normalize_a3rs:
            for a3r_r in a3r_rs:
                for aggregate in aggregates:
                    exp_index += 1
                    if a3r_r > 1 and normalize_base == 'StandardScaler':
                        logging.info('skipping experiment %s %s %s %s - incompatible (%d/%d)' % (normalize_base,
                                                                                                 normalize_a3r,
                                                                                                 a3r_r, aggregate,
                                                                                                 exp_index,
                                                                                                 total_experiments))
                        continue
                    logging.info('starting experiment %s %s %s %s (%d/%d)' % (normalize_base,
                                                                              normalize_a3r,
                                                                              a3r_r,
                                                                              aggregate,
                                                                              exp_index,
                                                                              total_experiments))

                    for task_id in all_task_ids:
                        openmldefaults.experiments.run_vanilla_surrogates_on_task(
                            task_id=task_id,
                            random_seed=args.random_seed,
                            search_space_identifier=args.search_space_identifier,
                            metadata_files=args.metadata_file,
                            scoring=args.scoring,
                            minimize_measure=args.minimize,
                            n_defaults=args.n_defaults,
                            n_configurations=args.n_configurations,
                            aggregate=aggregate,
                            a3r_r=a3r_r,
                            normalize_base=normalize_base,
                            normalize_a3r=normalize_a3r,
                            task_limit=args.task_limit,
                            surrogate_n_estimators=args.n_estimators,
                            surrogate_minimum_evals=args.minimum_evals,
                            output_directory=args.output_directory)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
