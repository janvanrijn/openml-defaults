import argparse
import numpy as np
import feather
import openmldefaults
import os
import pandas as pd
import pickle
import pulp
import time

from typing import List, Tuple

# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/mlr.classif.rpart.feather')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/openmldefaults')
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--resized_grid_size', type=int, default=5)
    parser.add_argument('--restricted_num_tasks', type=int, default=5)
    parser.add_argument('--num_defaults', type=int, default=2)
    return parser.parse_args()


def selected_set(df: pd.DataFrame, defaults: List[Tuple]):
    # filters out only the algorithms that we have in the 'set of defaults'
    df = df.loc[defaults]
    # df.min(axis=0) returns per dataset the minimum score obtained by 'set of defaults'
    # then we take the median of this
    return df.min(axis=0)


def reshape_configs(df, params, resized_grid_size):
    # subsample the hyperparameter grid
    for param in params:
        unique = np.array(df[param].unique())
        if len(unique) > resized_grid_size:
            interval = int(np.ceil(len(unique) / resized_grid_size))
            resized = unique[0::interval]
            assert len(resized) <= resized_grid_size, 'Param %s, originally %d, new size: %d' % (
            param, len(unique), len(resized))
            df = df.loc[df[param].isin(resized)]
    return df


def get_mixed_integer_formulation(df, num_defaults, num_tasks):
    big_m = num_tasks + 1  # keep the big m tight

    num_configurations, num_tasks = df.shape
    mip_optimizer = pulp.LpProblem('ComplementaryConfigurationSelector', pulp.LpMinimize)

    # creates a variable for each configuration (var = 1 iff it was selected to the default set)
    config_identifier_variables = list()
    for config_idx in range(num_configurations):
        current_dataset_min_score = pulp.LpVariable('config_' + str(df.index.tolist()[config_idx]), cat=pulp.LpBinary)
        config_identifier_variables.append(current_dataset_min_score)
    # ensures that only the required number of defaults is chosen
    mip_optimizer += pulp.lpSum(config_identifier_variables) == num_defaults

    # creates a variable for each dataset. This stores the minimal achievable score given the selected defaults
    dataset_min_score_variables = list()
    for task_idx, task_name in enumerate(df.columns):
        current_dataset_min_score = pulp.LpVariable(str(df.columns.values[task_idx]), 0.0, 1.0, pulp.LpContinuous)
        dataset_min_score_variables.append(current_dataset_min_score)

        # in order to store the minimum of a group of values (over which we minimize), we need a set of auxiliary
        # variables and big M conventions according to Greg Glockners answer on Stackoverlow
        # https://stackoverflow.com/questions/10792139/using-min-max-within-an-integer-linear-program
        auxilary_variables = list()
        for config_idx, civ in enumerate(config_identifier_variables):
            current_auxilary = pulp.LpVariable('task_%d_conf_%d' % (task_idx, config_idx), cat=pulp.LpBinary)
            mip_optimizer += current_dataset_min_score >= \
                             (df.iloc[config_idx][task_name] * config_identifier_variables[config_idx]) - \
                             (big_m * current_auxilary)
            # due to the nature of our problem, we need to ensure that the configuration variable OR the auxiliary
            # variable is set
            mip_optimizer += 0 <= 2 - config_identifier_variables[config_idx] - current_auxilary <= 1
            auxilary_variables.append(current_auxilary)
        mip_optimizer += pulp.lpSum(auxilary_variables) == num_configurations - 1

    # objective function: minimize the sum of score variables
    mip_optimizer += pulp.lpSum(dataset_min_score_variables)
    return mip_optimizer


def dominates(dominater, dominated):
    return sum([dominater[x] <= dominated[x] for x in range(len(dominater))]) == len(dominater)


def run(args):
    start_time = time.time()

    df = feather.read_dataframe(args.dataset_path)
    print(df.shape)

    if args.resized_grid_size is not None:
        df = reshape_configs(df, args.params, args.resized_grid_size)

    # always set the index
    df = df.set_index(args.params)

    if args.restricted_num_tasks is not None:
        # subsample num tasks
        df = df.iloc[:, 0:args.restricted_num_tasks]

    df, dominated = openmldefaults.utils.simple_cull(df, dominates)
    print('Dominated Configurations: %d/%d' % (len(dominated), len(df) + len(dominated)))
    mip_optimizer = get_mixed_integer_formulation(df, args.num_defaults, df.shape[1])
    if args.restricted_num_tasks is not None:
        experiment_dir = 'c%d_t%d_d%d' % (args.resized_grid_size, args.restricted_num_tasks, args.num_defaults)
    else:
        experiment_dir = 'c%d_tAll_d%d' % (args.resized_grid_size, args.num_defaults)
    os.makedirs(os.path.join(args.output_dir, experiment_dir), exist_ok=True)
    mip_optimizer.writeLP(os.path.join(args.output_dir, experiment_dir, 'minimize_multi_defaults.lp'))
    mip_optimizer.solve(solver=pulp.GLPK_CMD())
    run_time = time.time() - start_time

    # The status of the solution is printed to the screen
    result_dict = {'objective': pulp.value(mip_optimizer.objective),
                   'run_time': run_time,
                   'status': pulp.LpStatus[mip_optimizer.status]}

    # now gather the defaults
    defaults = []
    if result_dict['status'] == 'Optimal':
        # Each of the variables is printed with it's resolved optimum value
        for variable in mip_optimizer.variables():
            if variable.name.startswith('config_'):
                if variable.varValue == 1:
                    variables = variable.name.split('config_')[1][1:-1].split(',_')
                    variables = [float(var) for var in variables]
                    defaults.append(tuple(variables))

        # The optimised objective function value is printed to the screen
        result_dict['defaults'] = defaults
    print(result_dict)
    with open(os.path.join(args.output_dir, experiment_dir, 'results.pkl'), 'wb') as fp:
        pickle.dump(result_dict, fp)

    if result_dict['status'] == 'Optimal':
        result_frame = selected_set(df, defaults)
        diff = sum(result_frame) - result_dict['objective']
        assert abs(diff) < 0.00001, '%f vs %f' % (sum(result_frame), result_dict['objective'])


if __name__ == '__main__':
    run(parse_args())
