%load_ext autoreload
%autoreload
import warnings
import operator
from functools import partial, wraps

import deap.gp
import deap.tools
import numpy as np
import scipy.optimize

from glyph import gp
from glyph.utils import Memoize
from glyph.utils.numeric import nrmse, silent_numpy
import pdb


class Terminal(deap.gp.Terminal):
    def __init__(self, name):
        self.name = name
        self.ret = deap.gp.__type__

    def format(self, *args):
        return self.name


pset = gp.numpy_primitive_set(arity=0, categories=["algebraic"])
pset.terminals[object].append(Terminal("n"))
pset.terminals[object].append(Terminal("p"))



def phenotype(individual):
    pdb.set_trace()
    code = repr(individual)
    for i in range(individual.n_args):
       code = code.replace("p", "b{i}".format(i=i), 1)
    expr = "lambda {}: {}".format(args, code)
    func = eval(repr(individual), pset.context)
    return func


class Individual(gp.individual.AExpressionTree):
    """The gp representation (genotype) of the actuator for the control problem."""

    pset = pset

    @property
    def n_args(self):
        return sum([1 for t in self if isinstance(t, Terminal)])

    @property
    def arity(self):
        return 1


def const_opt(f, individual):
    arity = individual.arity

    @wraps(f)
    def closure(consts):
        new_consts = []
        for node in [
            consts[i : i + arity + 1] for i in range(0, len(consts), arity + 1)
        ]:
            new_consts.append(np.array(node[:-1]))
            new_consts.append(node[-1])
        return f(individual, *new_consts)

    p0 = np.ones((arity + 1) * individual.n_args)
    res = scipy.optimize.minimize(fun=closure, x0=p0, method="Nelder-Mead", tol=1e-2)
    popt = res.x if res.x.shape else np.array([res.x])
    measure_opt = res.fun
    if not res.success:
        warnings.warn(res.message, UserWarning)
    if measure_opt is None:
        measure_opt = closure(popt)
    return popt, measure_opt


@silent_numpy
def error(ind, *args):
    g = lambda x: x ** 2 - 1.1
    points = np.linspace(-1, 1, 100, endpoint=True).reshape(ind.arity, -1)
    y = g(points)
    f = phenotype(ind)
    args[0].dot(points) + args[1]  # ??? without this there will be segmentation fault
    yhat = f(points, *args).reshape(y.shape)

    return nrmse(y, yhat)


@Memoize
def measure(ind):
    popt, err_opt = const_opt(error, ind)
    ind.popt = popt
    return err_opt, len(ind)


def update_fitness(population, map=map):
    invalid = [p for p in population if not p.fitness.valid]
    fitnesses = map(measure, invalid)
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit
    return population

def static_limit(key, max_value):
    import copy
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                if key(ind) > max_value:
                    new_inds[i] = random.choice(keep_inds)
            return new_inds
        return wrapper
    return decorator

def main():
    pop_size = 20
    limit = static_limit(key=operator.attrgetter("height"), max_value=12)

    mate = limit(deap.gp.cxOnePoint)
    expr_mut = partial(deap.gp.genFull, min_=0, max_=2)
    mutate =   limit(partial(deap.gp.mutUniform, expr=expr_mut, pset=Individual.pset))
    select =   partial(deap.tools.selDoubleTournament, fitness_size=10, parsimony_size=1.6, fitness_first=True)

    algorithm = gp.algorithms.AgeFitness(
        mate, mutate, select, Individual.create_population
    )

    pop = update_fitness(Individual.create_population(pop_size))

    for gen in range(20):
        pop = algorithm.evolve(pop)
        pop = update_fitness(pop)
        best = deap.tools.selBest(pop, 1)[0]

        print(best, best.popt, best.fitness.values)

        if best.fitness.values[0] <= 1e-3:
            break


if __name__ == "__main__":
    main()





