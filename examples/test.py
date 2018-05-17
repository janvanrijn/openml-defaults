import numpy as np
import pulp

prob = pulp.LpProblem('Multi min', pulp.LpMinimize)

problem_size = 160
vals = np.random.random_sample(problem_size)
print(np.min(vals), vals)

z = pulp.LpVariable("z")
# binary auxilary var
b = [pulp.LpVariable("b%d" % i, cat=pulp.LpBinary) for i in range(problem_size)]

# the var that determines which to choise
c = [pulp.LpVariable("c%d" % i, cat=pulp.LpBinary) for i in range(problem_size)]


# big M
M = 100

num_c = 1

# The objective function is added to 'prob' first
prob += z

for i in range(problem_size):
    prob += z >= (vals[i] * c[i]) - (M * b[i])

prob += sum(c) == num_c
prob += sum(b) == problem_size - 1

for i in range(problem_size):
    prob += 0 <= 2 - c[i] - b[i] <= 1

prob.writeLP("minimize_multi.lp")

# The problem is solved using PuLP's choice of Solver
prob.solve()

# The status of the solution is printed to the screen
print("Status:", pulp.LpStatus[prob.status])

# Each of the variables is printed with it's resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)

# The optimised objective function value is printed to the screen
print("Outcome = ", pulp.value(prob.objective))

assert np.abs(pulp.value(prob.objective) - np.min(vals)) < 0.00001

