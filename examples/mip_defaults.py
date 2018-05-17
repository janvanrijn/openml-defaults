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
    parser.add_argument('--restricted_num_tasks', type=int, default=1)
    parser.add_argument('--num_defaults', type=int, default=1)
    return parser.parse_args()


def run(args):
    big_m = 100

    df = feather.read_dataframe(args.dataset_path)

    if args.resized_grid_size is not None:
        # subsample the hyperparameter grid
        for param in args.params:
            unique = np.array(df[param].unique())
            interval = int(np.ceil(len(unique) / args.resized_grid_size))
            resized = unique[0::interval]
            assert len(resized) == args.resized_grid_size
            df = df.loc[df[param].isin(resized)]
    # always set the index
    df = df.set_index(args.params)
    if args.restricted_num_tasks is not None:
        # subsample num tasks
        df = df.iloc[:, 0:args.restricted_num_tasks]

    print(df)
    num_configurations, num_tasks = df.shape

    mip_optimizer = pulp.LpProblem('ComplementaryConfigurationSelector', pulp.LpMinimize)

    config_identifier_variables = list()
    for config_idx in range(len(df)):
        current_dataset_min_score = pulp.LpVariable(str(df.index.tolist()[config_idx]), cat=pulp.LpBinary)
        config_identifier_variables.append(current_dataset_min_score)
    mip_optimizer += pulp.lpSum(config_identifier_variables) == args.num_defaults

    dataset_min_score_variables = list()
    for task_idx, task_name in enumerate(df.columns):
        current_dataset_min_score = pulp.LpVariable(str(df.columns.values[task_idx]), 0.0, 1.0, pulp.LpContinuous)
        dataset_min_score_variables.append(current_dataset_min_score)

        # ensure that current is the min error rate from all selected configurations
        auxilary_variables = list()
        for config_idx, civ in enumerate(config_identifier_variables):
            current_auxilary = pulp.LpVariable('task_%d_conf_%d' % (task_idx, config_idx), cat=pulp.LpBinary)
            mip_optimizer += current_dataset_min_score <= (df.iloc[config_idx][task_name] * config_identifier_variables[config_idx]) - (big_m * current_auxilary)
            mip_optimizer += 0 <= 2 - config_identifier_variables[config_idx] - current_auxilary <= 1
            auxilary_variables.append(current_auxilary)
        mip_optimizer += pulp.lpSum(auxilary_variables) == num_configurations - 1

    mip_optimizer += pulp.lpSum(dataset_min_score_variables)

    mip_optimizer.solve()

    # The status of the solution is printed to the screen
    status = pulp.LpStatus[mip_optimizer.status]
    print("Status:", status)

    if status == 'Optimal':
        # Each of the variables is printed with it's resolved optimum value
        for v in mip_optimizer.variables():
            print(v.name, "=", v.varValue)

        # The optimised objective function value is printed to the screen
        print("Outcome = ", pulp.value(mip_optimizer.objective))


if __name__ == '__main__':
    run(parse_args())
