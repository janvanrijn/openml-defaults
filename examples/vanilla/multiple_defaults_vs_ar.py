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
    metadata_file_text_classification = os.path.expanduser('../../data/text_classification.arff')
    parser = argparse.ArgumentParser(description='Creates an ARFF file')
    parser.add_argument('--output_directory', type=str, help='directory to store output',
                        default=os.path.expanduser('~/experiments/openml-defaults/at_vs_ar/'))
    parser.add_argument('--task_idx', type=int)
    parser.add_argument('--metadata_files', type=str, nargs='+', default=[metadata_file_text_classification])
    parser.add_argument('--scoring', type=str, default='missclassification_rate')
    parser.add_argument('--search_space_identifier', type=str, default='ferreira')
    parser.add_argument('--minimize', action='store_true', default=True)
    parser.add_argument('--normalize_base', type=str, default=None)
    parser.add_argument('--normalize_a3r', type=str, default=None)
    parser.add_argument('--a3r_r', type=int, default=2)
    parser.add_argument('--aggregate', type=str, choices=openmldefaults.experiments.AGGREGATES, default='sum')
    parser.add_argument('--n_defaults', type=int, default=384)
    parser.add_argument('--n_estimators', type=int, default=64)
    parser.add_argument('--minimum_evals', type=int, default=128)
    parser.add_argument('--random_iterations', type=int, default=1)
    parser.add_argument('--run_on_surrogates', action='store_true', default=True)
    parser.add_argument('--task_limit', type=int, default=None, help='For speed')
    parser.add_argument('--task_id_column', default='dataset', type=str)
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

        for task_id in task_ids_to_process:
            openmldefaults.experiments.run_vanilla_surrogates_on_task(
                task_id=task_id,
                models=[openmldefaults.models.AverageRankDefaults(), openmldefaults.models.ActiveTestingDefaults()],
                use_surrogates=False,
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
                runtime_column='runtime',
                consider_a3r=True,
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
