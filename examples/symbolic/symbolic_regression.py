from deap import gp
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import operator
import random
import math
import numpy as np
from sympy import simplify

def optim_func(x):
    return x**4 - x**3 - x**2 - x

def optim_func(x):
    return 0.66

# -- Primitives --------------------------------------------------------------------------
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def protectedPow(left, right):
    try:
        x = pow(left, right)
        if isinstance(x, complex) or x < -10e10 or x > 10e10:
            raise ValueError()
        return x
    except (ValueError, OverflowError, ZeroDivisionError):
        return 1

# Operations
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2, name = "div")
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(protectedPow, 2, name = "pow")
pset.addEphemeralConstant("randu0e10", lambda: random.uniform(0,1))
pset.addEphemeralConstant("int15", lambda: random.randint(0,10))
pset.renameArguments(ARG0='x')


# Each Individual is a Tree which aims to maximize its negative fitness.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


# Initialize the population.
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Objective
def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - optim_func(x))**2 for x in points)
    return math.fsum(sqerrors) / len(points),

# Evaluation:
toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])

# Selection / Mutation Operations
toolbox.register("select", tools.selDoubleTournament, fitness_size=10, parsimony_size=1.6, fitness_first=True)
toolbox.register("mate", gp.cxOnePoint)

def random_mutation_operator(individual):
    '''
        Randomly picks a replacement, insert, or shrink mutation.
    '''
    roll = random.random()
    if roll <= 0.25:
        return gp.mutUniform(individual, expr=toolbox.expr_mut, pset=pset)
    elif roll <= 0.5:
        return gp.mutInsert(individual, pset=pset)
    elif roll <= 0.6:
        return gp.mutEphemeral(individual, mode = "one")
    else:
        return gp.mutShrink(individual)

toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register('mutate', random_mutation_operator)
toolbox.register("mutate", random_mutation_operator)



# Limit complexity
toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))

# Statistics
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.5, 100, stats=mstats, halloffame=hof, verbose=True)

best_ind = tools.selBest(pop, 1)[0]
print("Best individual is %s" % (best_ind))


def simplify_expression(ind):
    from numpy import subtract as sub
    from numpy import multiply as mul
    from numpy import true_divide as div
    from numpy import negative as neg
    from numpy import add, square
    from sympy import symbols, Function
    pow = symbols('pow', cls=Function)
    x = symbols('x')
    exec("expr = %s" % str(ind))
    return(expr)

def draw_graph(ind):
    import pygraphviz as pgv
    nodes, edges, labels = gp.graph(ind)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw("tree.pdf")



if False:
    simplify_expression(best_ind)
    draw_graph(best_ind)

