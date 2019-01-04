import argparse
import logging
import openml
import openmldefaults
import os
import pandas as pd


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/dsi/users/jv2657/experiments ~/habanero_experiments
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
    parser.add_argument('--normalize_base', type=str, default=None)
    parser.add_argument('--normalize_a3r', type=str, default='StandardScaler')
    parser.add_argument('--a3r_r', type=int, default=1)
    parser.add_argument('--aggregate', type=str, choices=openmldefaults.experiments.AGGREGATES, default='sum')
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
    if args.task_idx is None:
        all_task_ids = study.tasks
    else:
        all_task_ids = [study.tasks[args.task_idx]]

    for task_id in all_task_ids:
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
            aggregate=args.aggregate,
            a3r_r=args.a3r_r,
            normalize_base=args.normalize_base,
            normalize_a3r=args.normalize_a3r,
            task_limit=args.task_limit,
            output_directory=args.output_directory)


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
