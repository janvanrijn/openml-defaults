from deap import gp
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random

def random_mutation_operator(individual):
    '''
        Randomly picks a replacement, insert, mutEphemeral or shrink mutation.
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


def eval_squared_error(optim_func, individual, points, alpha = 0.01):
    '''
        Defines an objective: The squared difference between optim_func and the evaluated individual for
        all points.

        optim_func: A function to be optimized
        individual: A individual from the population (gp.PrimitiveTree)
        points: Points to evaluate optim_func and the individual at.
        alpha: Complexity penalty for length of the individual.
    '''
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - optim_func(x))**2 for x in points)
    out = math.fsum(sqerrors) / len(points)
    out += alpha * max(0, len(individual) - 8)**2
    return out,


def optimize_function(objective, pset, pars, points = [x/10. for x in range(-10,10)]):

    # Each Individual is a Tree which aims to maximize its negative fitness.
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    # Initialize the population.
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=0, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)


    # Evaluation:
    toolbox.register("evaluate", objective, points=points)

    # Selection / Mutation Operations
    toolbox.register("select", tools.selDoubleTournament, fitness_size=10, parsimony_size=1.6, fitness_first=True)
    toolbox.register("mate", gp.cxOnePoint)

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
    return objective(best_ind, points=points), str(best_ind)
