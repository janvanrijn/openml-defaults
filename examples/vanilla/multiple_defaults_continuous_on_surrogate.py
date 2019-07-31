import arff
import argparse
import json
import logging
import numpy as np
import openmlcontrib
import openmldefaults
import os
import pandas as pd


# SSHFS NEMO FREIBURG:
# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
#
# SSHFS GRACE LEIDEN:
# ssh -f -N -L 1233:grace.liacs.nl:22 rijnjnvan@gold.liacs.nl
# sshfs -p 1233 vanrijn@localhost:/home/vanrijn/experiments ~/grace_experiments
def parse_args():
    metadata_file_svc = os.path.expanduser('~/data/openml-defaults/svc.arff')
    metadata_file_gb = os.path.expanduser('~/data/openml-defaults/gradient_boosting.arff')
    metadata_file_adaboost019 = os.path.expanduser('~/projects/openml-pimp/KDD2018/data/arff/adaboost.arff')
    metadata_file_random_forest019 = os.path.expanduser('~/projects/openml-pimp/KDD2018/data/arff/random_forest.arff')
    metadata_file_svc019 = os.path.expanduser('~/projects/openml-pimp/KDD2018/data/arff/svc.arff')
    metadata_file_resnet = os.path.expanduser('~/projects/hypeCNN/data/12param/resnet.arff')
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~') + '/experiments/openml-defaults/vanilla_defaults_vs_rs/')
    parser.add_argument('--task_idx', type=int, default=0)
    parser.add_argument('--metadata_files', type=str, nargs='+', default=[metadata_file_adaboost019, metadata_file_random_forest019, metadata_file_svc019])
    parser.add_argument('--scoring', type=str, default='predictive_accuracy')
    parser.add_argument('--search_space_identifier', type=str, default=None)
    parser.add_argument('--minimize', action='store_true')
    parser.add_argument('--normalize_base', type=str, default='MinMaxScaler')
    parser.add_argument('--normalize_a3r', type=str, default='StandardScaler')
    parser.add_argument('--a3r_r', type=int, default=1)
    parser.add_argument('--aggregate', type=str, choices=openmldefaults.experiments.AGGREGATES, default='sum')
    parser.add_argument('--n_defaults', type=int, default=32)
    parser.add_argument('--n_estimators', type=int, default=64)
    parser.add_argument('--minimum_evals', type=int, default=128)
    parser.add_argument('--random_iterations', type=int, default=1)
    parser.add_argument('--run_on_surrogates', action='store_true')
    parser.add_argument('--task_limit', type=int, default=None, help='For speed')
    parser.add_argument('--task_id_column', default='task_id', type=str)
    parser.add_argument('--override_parameters', type=str)
    args_ = parser.parse_args()
    return args_


def run(args):
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    task_ids = None
    for arff_file in args.metadata_files:
        with open(arff_file, 'r') as fp:
            df = openmlcontrib.meta.arff_to_dataframe(arff.load(fp), None)
            if task_ids is None:
                task_ids = np.sort(np.unique(df[args.task_id_column].values))
            else:
                task_ids = np.sort(np.unique(np.append(task_ids, df[args.task_id_column].values)))
    logging.info('Task ids: %s' % task_ids)
    if args.task_idx is None:
        task_ids_to_process = task_ids
    else:
        task_ids_to_process = [task_ids[args.task_idx]]

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
            run_on_surrogate=args.run_on_surrogates,
            output_directory=args.output_directory,
            task_id_column=args.task_id_column,
            skip_row_check=True,
            override_parameters=json.loads(args.override_parameters) if args.override_parameters else None,
        )

        for task_id in task_ids_to_process:
            openmldefaults.experiments.run_vanilla_surrogates_on_task(
                task_id=task_id,
                models=[openmldefaults.models.GreedyDefaults()],
                use_surrogates=True,
                random_seed=random_seed,
                search_space_identifier=args.search_space_identifier,
                metadata_files=args.metadata_files,
                scoring=args.scoring,
                minimize_measure=args.minimize,
                n_defaults=args.n_defaults,
                aggregate=args.aggregate,
                a3r_r=args.a3r_r,
                normalize_base=args.normalize_base,
                normalize_a3r=args.normalize_a3r,
                surrogate_n_estimators=args.n_estimators,
                surrogate_minimum_evals=args.minimum_evals,
                runtime_column=None,
                consider_a3r=False,
                run_on_surrogate=args.run_on_surrogates,
                task_limit=args.task_limit,
                output_directory=args.output_directory,
                task_id_column=args.task_id_column,
                skip_row_check=True,
                override_parameters=json.loads(args.override_parameters) if args.override_parameters else None
            )


if __name__ == '__main__':
    pd.options.mode.chained_assignment = 'raise'
    run(parse_args())
