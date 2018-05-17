import argparse
import numpy as np
import feather
import os
import pandas as pd
import pulp

from typing import List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/mlr.classif.rpart.feather')
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--resized_grid_size', type=int, default=2)
    parser.add_argument('--restricted_num_tasks', type=int, default=5)
    parser.add_argument('--num_defaults', type=int, default=1)
    return parser.parse_args()


def run(args):
    big_m = 100
    epsilon = 0.1

    df = feather.read_dataframe(args.dataset_path)

    if args.resized_grid_size is not None:
        # subsample the hyperparameter grid
        for param in args.params:
            unique = np.array(df[param].unique())
            interval = int(np.ceil(len(unique) / args.resized_grid_size))
            resized = unique[0::interval]
            assert len(resized) == args.resized_grid_size
            df = df.loc[df[param].isin(resized)]
    if args.restricted_num_tasks is not None:
        df = df.iloc[:, 0:args.restricted_num_tasks]

    df = df.set_index(args.params)

    mip_optimizer = pulp.LpProblem('ComplementaryConfigurationSelector', pulp.LpMinimize)

    config_identifier_variables = list()
    for config_idx in range(len(df)):
        current = pulp.LpVariable(str(df.index.tolist()[config_idx]), 0, 1, pulp.LpBinary)
        config_identifier_variables.append(current)
    mip_optimizer += pulp.lpSum(config_identifier_variables) == args.num_defaults

    dataset_min_score_variables = list()
    for task_idx, task_name in enumerate(df.columns):
        current = pulp.LpVariable(str(df.columns.values[task_idx]), 0, 1, pulp.LpContinuous)
        dataset_min_score_variables.append(current)

        # ensure that current is the min error rate from all selected configurations
        for config_idx, civ in enumerate(config_identifier_variables):
            mip_optimizer += current <= df.iloc[config_idx][task_name] * config_identifier_variables[config_idx]

    mip_optimizer += sum([dmsc for dmsc in dataset_min_score_variables])

    mip_optimizer.solve()

    # Each of the variables is printed with it's resolved optimum value
    for v in mip_optimizer.variables():
        print(v.name, "=", v.varValue)


if __name__ == '__main__':
    run(parse_args())
