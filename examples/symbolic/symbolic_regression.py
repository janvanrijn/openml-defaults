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
from mlflow import log_metric, log_param, log_artifact

def optimize_function(optim_func, pset, pars, points = [x/10. for x in range(-10,10)]):

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
    def evalSymbReg(individual, points, alpha = 0.01):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        # and the real function : x**4 + x**3 + x**2 + x
        sqerrors = ((func(x) - optim_func(x))**2 for x in points)
        out = math.fsum(sqerrors) / len(points)
        out += alpha * max(0, len(individual) - 8)**2
        return out,

    # Evaluation:
    toolbox.register("evaluate", evalSymbReg, points=points)

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
        elif roll <= 0.7:
            return gp.mutEphemeral(individual, mode = "one")
        else:
            return gp.mutShrink(individual)

    toolbox.register("expr_mut", gp.genGrow, min_=0, max_=3)
    toolbox.register('mutate', random_mutation_operator)
    # Limit complexity
    toolbox.decorate("mate",   gp.staticLimit(key=operator.attrgetter("height"), max_value=6))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=6))

    mstats = register_stats()
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, pars["cxpb"], pars["mutpb"], 100, stats=mstats, halloffame=hof, verbose=True)

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s" % str(best_ind))
    return evalSymbReg(best_ind, points=points), str(best_ind)


def register_pset():
    np.seterr(all='raise')
    # -- Primitives --------------------------------------------------------------------------
    def protectedDiv(left, right):
        try:
            return np.divide(left, right)
        except (ZeroDivisionError, FloatingPointError):
            return 1.

    def protectedPow(left, right):
        try:
            x = pow(left, right)
            if isinstance(x, complex) or x < -10e10 or x > 10e10:
                raise ValueError()
            return round(x, 16)
        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
            return 1.

    def protectedExp(x):
        try:
            x = np.exp(x)
            if (x > 100):
                raise ValueError()
            return(round(x, 16))
        except:
            return(1)

    def sqrt_abs(x):
        if (x < 0):
            x = abs(x)
        x = np.sqrt(x)
        return(x)

    # Operations
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2, name = "div")
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(sqrt_abs, 1, name = "sqrt")
    # pset.addPrimitive(log, 1, name = "log")
    pset.addPrimitive(protectedPow, 2, name = "pow")
    pset.addEphemeralConstant("rand01", lambda: random.uniform(0,1))
    pset.addEphemeralConstant("int010", lambda: random.randint(1, 10))
    pset.addEphemeralConstant("pow10", lambda: protectedPow(10, random.randint(-3, 3)))
    pset.renameArguments(ARG0='x')
    return(pset)

def register_stats():
    # Statistics
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return mstats

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



# if False:
#     simplify_expression(best_ind)
#     draw_graph(best_ind)


if __name__ == '__main__':
    # def optim_func(x):
    #     return x**4 - x**3 - x**2 - x
    # optimize_function(optim_func)

    pset = register_pset()

    pars = {
        "cxpb"  : 0.7, # Cross-Over Probability
        "mutpb" : 0.7  # Mutation   Probability
    }

    def optim_func(x):
        return np.sqrt(x) - 1.1

    performance, best = optimize_function(optim_func, pset, pars, points=[x/10. for x in range(1,100)])

    # Log MLFlow
    [log_param(x, pars[x]) for x in pars.keys()]
    log_metric("performance", performance[0])
