import argparse
import numpy as np
import pulp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rows', type=int, default=256)
    parser.add_argument('--num_cols', type=int, default=25)
    parser.add_argument('--select_rows', type=int, default=1)
    parser.add_argument('--solver', type=str, default='GLPK_CMD')
    return parser.parse_args()


def sum_of_mins(matrix, rows):
    return sum(np.amin(matrix[rows], axis=0))


if __name__ == '__main__':
    args = parse_args()
    big_m = args.num_cols + 1

    matrix = np.random.rand(args.num_rows, args.num_cols)

    pulp_optimizer = pulp.LpProblem('Minimize sum of columns', pulp.LpMinimize)

    # creates a variable for each configuration (var = 1 iff it was selected to the default set)
    row_identifier_vars = list()
    for row_idx in range(args.num_rows):
        row_identifier = pulp.LpVariable('row_%s' % row_idx, cat=pulp.LpBinary)
        row_identifier_vars.append(row_identifier)
    # ensures that only the required number of rows is chosen
    pulp_optimizer += pulp.lpSum(row_identifier_vars) == args.select_rows

    # creates a variable for each column. This stores the minimal achievable score given the selected rows
    col_min_value_vars = list()
    for col_idx in range(args.num_cols):
        current_col_min_value = pulp.LpVariable('col_%s' % col_idx, 0, 1.0, pulp.LpContinuous)
        col_min_value_vars.append(current_col_min_value)

        # in order to store the minimum of a group of values (over which we minimize), we need a set of auxiliary
        # variables and big M conventions according to Greg Glockners answer on Stackoverlow
        # https://stackoverflow.com/questions/10792139/using-min-max-within-an-integer-linear-program
        auxilary_variables = list()
        for row_idx in range(args.num_rows):
            current_auxilary = pulp.LpVariable('aux_col_%d_row_%d' % (col_idx, row_idx), cat=pulp.LpBinary)
            pulp_optimizer += current_col_min_value >= (matrix[row_idx][col_idx] * row_identifier_vars[row_idx]) - (big_m * current_auxilary)
            # due to the nature of our problem, we need to ensure that the configuration variable OR the auxiliary
            # variable is set
            pulp_optimizer += 0 <= 2 - row_identifier_vars[row_idx] - current_auxilary <= 1
            auxilary_variables.append(current_auxilary)
        pulp_optimizer += pulp.lpSum(auxilary_variables) == args.num_rows - 1

    # objective function: minimize the sum of score variables
    pulp_optimizer += pulp.lpSum(col_min_value_vars)

    pulp_optimizer.solve(solver=getattr(pulp, args.solver)())

    selected_rows = []
    if pulp.LpStatus[pulp_optimizer.status] == 'Optimal':
        for variable in pulp_optimizer.variables():
            if variable.name.startswith('row_'):
                if variable.varValue == 1:
                    selected_rows.append(int(variable.name.split('row_')[1]))

    difference = abs(pulp.value(pulp_optimizer.objective) - sum_of_mins(matrix, selected_rows))
    assert difference < 0.00001
