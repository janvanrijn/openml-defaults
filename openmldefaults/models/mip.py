import numpy as np
import openmldefaults
import pulp
import time
import typing


class MipDefaults(object):

    def __init__(self, solver):
        self.name = 'mip'
        self.solver = solver

    @staticmethod
    def get_mixed_integer_formulation(df, num_defaults) -> typing.Tuple[typing.List, typing.Dict[str, typing.Any]]:
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

        # make sure that the dataframe has positive numbers only
        df_min = df.min().min()
        if df_min > 0:
            df_min = 0
        mip_optimizer += pulp.lpSum(y[row][column] * (df.iloc[row, column] + (-1 * df_min))
                                    for row in range(num_configurations) for column in range(num_tasks))
        return mip_optimizer

    def generate_defaults(self, df, num_defaults):
        mip_optimizer = MipDefaults.get_mixed_integer_formulation(df, num_defaults)

        start_time = time.time()
        mip_optimizer.solve(solver=getattr(pulp, self.solver)())
        runtime = time.time() - start_time

        if pulp.LpStatus[mip_optimizer.status] != 'Optimal':
            raise Exception('Solver had error: %s' % pulp.LpStatus[mip_optimizer.status])

        # now gather the defaults
        selected_indices = []
        # Each of the variables is printed with it's resolved optimum value
        for variable in mip_optimizer.variables():
            if variable.name.startswith('config_'):
                if variable.varValue == 1:
                    index = int(variable.name.split('config_')[1])
                    config = df.index.values[index]
                    selected_indices.append(config)

        print(openmldefaults.utils.get_time(), selected_indices)

        # we must also recalculate the objective, due to the scaling
        result_frame = openmldefaults.utils.selected_set(df, selected_indices)

        results_dict = {
            'objective': sum(result_frame),
            'run_time': runtime,
        }
        return selected_indices, results_dict
