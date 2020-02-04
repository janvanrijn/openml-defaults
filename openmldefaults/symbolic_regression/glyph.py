import copy
import warnings
import operator
from functools import partial, wraps, partialmethod

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

def make_pset(meta_features:list=["n", "p"]):
    """
    Create a pset

    :param meta_features: list(str): A list of meta-feature names.

    :return: Returns a set of primitives [deap.gp.PrimitiveSet]
    """
    pset = gp.numpy_primitive_set(arity=0, categories=["algebraic"])
    pset.terminals[object].append(Terminal("const"))
    for ft in meta_features:
        pset.terminals[object].append(Terminal(ft))
    pset.meta_features = meta_features
    return pset

def static_limit(key, max_value):
    """
    Wraps a gp.deap mate or mutate expression, enforcing static maximum height of
    the AExpressionTree's
    :param max_value: Maximum depth of the expression tree.
    :return: Wrapped function
    """

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

def phenotype(individual):
    """
        Separates out constants, i.e. "const + const" -> "c1 + c2"
        for individual optimization using `scipy.optimize.minimize`.

        :param individual: [AExpressionTree] Individual to be converted.

        :return: Function to be optimized,
    """
    iargs = ""
    if (individual.n_consts > 0):
        iargs = iargs+",".join("c{i}".format(i=i) for i in range(individual.n_consts))
    if (len(individual.pset.meta_features) > 0):
        iargs = ",".join([iargs, ",".join(mf for mf in individual.pset.meta_features)])
    code = repr(individual)
    for i in range(individual.n_consts):
       code = code.replace("const", "c{i}".format(i=i), 1)
    code = code.replace("'", "")
    expr = "lambda {}: {}".format(iargs, code)
    func = eval(expr, individual.pset.context)
    return func

# ------ 1-D Individual ---------------------------------------------------------
class Individual(gp.individual.AExpressionTree):
    """The gp representation (genotype) of the formula"""
    pset = make_pset()
    name = "1-D formula"
    @property
    def n_consts(self):
        return sum([1 for t in self if isinstance(t, Terminal) if t.name == "const"])
    @property
    def n_args(self):
        return sum([1 for t in self if isinstance(t, Terminal)])
    @property
    def arity(self):
        return 1

# ------ N-D Individual ---------------------------------------------------------
class NDIndividual(gp.individual.ANDimTree):
    """The gp representation (genotype) of a N-Dimensional formula"""
    base = Individual
    name = "N-D formula"
    @property
    def n_consts(self):
        return sum([self[i].n_consts for i in range(self.height)])
    @property
    def n_args(self):
        return sum([self[i].n_args for i in range(self.height)])
    @property
    def arity(self):
        return sum([self[i].arity for i in range(self.height)])
    @property
    def pset(self):
        return self[0].pset


def const_opt(f, individual, meta_features={"n":10,"p":4}):
    arity = individual.arity
    f = partial(f, **meta_features)

    @wraps(f)
    def closure(consts):
        consts = dict([("c"+"{i}".format(i = k), v) for k,v in enumerate(consts)])
        params = f(individual, **consts)
        return surrogate_prd(params)

    p0 = np.ones(individual.n_consts)
    res = scipy.optimize.minimize(fun=closure, x0=p0, method="Nelder-Mead", tol=1e-2, options={"maxiter":30})
    popt = res.x if res.x.shape else np.array([res.x])
    measure_opt = res.fun
    if not res.success:
        warnings.warn(res.message, UserWarning)
    if measure_opt is None:
        measure_opt = closure(popt)
    return popt, measure_opt


def surrogate_prd(x):
    return x[0] - 10 + x[1]+3


@silent_numpy
def error(ind, **args):
    f = phenotype(ind)
    return f(**args)

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

if __name__ == "__main__":
    pop_size = 20
    # Koza-style tree depth limits.
    limit = static_limit(key=operator.attrgetter("height"), max_value=12)

    # Mutations
    mate = limit(deap.gp.cxOnePoint)
    expr_mut = partial(deap.gp.genFull, min_=0, max_=2)
    mutate =   limit(partial(deap.gp.mutUniform, expr=expr_mut, pset=Individual.pset))
    select =   partial(deap.tools.selDoubleTournament, fitness_size=10, parsimony_size=1.6, fitness_first=True)

    algorithm = gp.algorithms.AgeFitness(
        mate, mutate, select, NDIndividual.create_population
    )

    pop = update_fitness(NDIndividual.create_population(pop_size, 2))

    for gen in range(20):
        pop = algorithm.evolve(pop)
        pop = update_fitness(pop)
        best = deap.tools.selBest(pop, 1)[0]

        print(best, best.popt, best.fitness.values)

        if best.fitness.values[0] <= 1e-3:
            break






