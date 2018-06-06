import argparse
import json
import numpy as np
import openmldefaults
import os
import pickle
import pulp
import time


# sshfs jv2657@habanero.rcs.columbia.edu:/rigel/home/jv2657/experiments ~/habanero_experiments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.expanduser('~') + '/data/openml-defaults/train_svm.feather')
    parser.add_argument('--flip_performances', action='store_true')
    parser.add_argument('--output_dir', type=str, default=os.path.expanduser('~') + '/experiments/openml-defaults')
    parser.add_argument('--params', type=str, nargs='+', required=True)
    parser.add_argument('--resized_grid_size', type=int, default=10)
    parser.add_argument('--num_defaults', type=int, default=5)
    parser.add_argument('--solver', type=str, default='GLPK_CMD')
    return parser.parse_args()


def get_mixed_integer_formulation(df, num_defaults):
    num_configurations, num_tasks = df.shape
    mip_optimizer = pulp.LpProblem('ComplementaryConfigurationSelector', pulp.LpMinimize)

    Q = list()
    for row in range(num_configurations):
        Q.append(list())
        for column in range(num_tasks):
            Q[row].append(set())
            for row_prime in range(num_configurations):
                if df.iloc[row_prime, column] < df.iloc[row, column]:
                    Q[row][column].add(row_prime)
                elif df.iloc[row_prime, column] == df.iloc[row, column] and row_prime < row:
                    Q[row][column].add(row_prime)

    x = np.empty(num_configurations, dtype=pulp.LpVariable)
    for row in range(num_configurations):
        current = pulp.LpVariable("config_%d" % row, cat=pulp.LpBinary)
        x[row] = current
    mip_optimizer += pulp.lpSum(x) == num_defaults

    y = np.empty((num_configurations, num_tasks), dtype=pulp.LpVariable)
    for row in range(num_configurations):
        for column in range(num_tasks):
            current = pulp.LpVariable('auxilary_%d_%d' % (row, column), cat=pulp.LpContinuous)
            x_vars = [x[s] for s in Q[row][column]]
            y[row, column] = current
            mip_optimizer += current >= x[row] - pulp.lpSum(x_vars)
            mip_optimizer += current >= 0

    mip_optimizer += pulp.lpSum(y[row][column] * df.iloc[row, column] for row in range(num_configurations) for column in range(num_tasks))
    return mip_optimizer


def run(args):

    df = openmldefaults.utils.load_dataset(args.dataset_path, args.params, args.resized_grid_size, args.flip_performances)
    if df.min().min() < 0:
        df[df.columns.values] += (-1 * df.min().min())
    assert(df.min().min() >= 0)

    mip_optimizer = get_mixed_integer_formulation(df, args.num_defaults)

    solver_dir = 'mip_%s' % args.solver
    experiment_dir = openmldefaults.utils.get_setup_dirname(args.resized_grid_size, args.num_defaults)
    os.makedirs(os.path.join(args.output_dir, solver_dir, experiment_dir), exist_ok=True)
    outputfile = os.path.join(args.output_dir, solver_dir, experiment_dir, 'minimize_multi_defaults.lp')
    mip_optimizer.writeLP(outputfile)
    start_time = time.time()
    mip_optimizer.solve(solver=getattr(pulp, args.solver)())
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
                    index = int(variable.name.split('config_')[1])
                    config = df.index.values[index]
                    defaults.append(config)

        # The optimised objective function value is printed to the screen
        result_dict['defaults'] = defaults
    print(result_dict)
    with open(os.path.join(args.output_dir, solver_dir, experiment_dir, 'results.pkl'), 'wb') as fp:
        pickle.dump(result_dict, fp)

    if result_dict['status'] == 'Optimal':
        result_frame = openmldefaults.utils.selected_set(df, defaults)
        diff = sum(result_frame) - result_dict['objective']
        assert abs(diff) < 0.00001, '%f vs %f' % (sum(result_frame), result_dict['objective'])
    else:
        raise Exception('Exit with status: %s' % result_dict['status'])


if __name__ == '__main__':
    run(parse_args())
