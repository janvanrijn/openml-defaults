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
from glyph.gp.breeding import nd_crossover, nd_mutation
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

        This builds a lambda function of the format
        lambda a,b,c0,c1: [Add(a, c0), Add(a,b)]

        where c0, c1 are constants to be optimized
        and   a , b  are meta-feature constants.
        The function returns a list of length *ndim*.

        :param individual: [AExpressionTree] Individual to be converted.

        :return: Function to be optimized.

    """
    # String of args to the lambda function:
    if (individual.n_consts > 0):
        iargs = ",".join("c{i}".format(i=i) for i in range(individual.n_consts))
    if (len(individual.pset.meta_features) > 0):
        mfargs = ",".join(mf for mf in individual.pset.meta_features)
        if (individual.n_consts > 0):
            iargs = ",".join([iargs, mfargs])
        else:
            iargs = mfargs
    # Build up the lambda function and evaluate it
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


def const_opt(f, individual, mfargs):
    arity = individual.arity
    f = partial(f, **mfargs)

    @wraps(f)
    def closure(consts):
        if consts is not None:
            consts = dict([("c"+"{i}".format(i = k), v) for k,v in enumerate(consts)])
            params = f(individual, **consts)
        else:
            params = f(individual)
        return surrogate_prd(params)


    if (individual.n_consts > 0):
        p0 = np.ones(individual.n_consts)
        res = scipy.optimize.minimize(fun=closure, x0=p0, method="Nelder-Mead", tol=1e-2, options={"maxiter":30})
        popt = res.x if res.x.shape else np.array([res.x])
        measure_opt = res.fun
        if not res.success:
            warnings.warn(res.message, UserWarning)
        if measure_opt is None:
            measure_opt = closure(popt)
    else:
        measure_opt = closure(None)
        popt = np.array([])
    return popt, measure_opt


def surrogate_prd(x):
    return (x[0] - 10)**2 + (x[1]+3)**2


@silent_numpy
def error(ind, **args):
    f = phenotype(ind)
    return f(**args)

@Memoize
def measure(ind, mfargs):
    popt, err_opt = const_opt(f=error, individual=ind, mfargs=mfargs)
    ind.popt = popt
    ind_len = sum([len(x) for x in ind])
    return err_opt, ind_len


def update_fitness(population, mfargs, map=map):
    invalid = [p for p in population if not p.fitness.valid]
    fitnesses = map(partial(measure, mfargs=mfargs), invalid)
    for ind, fit in zip(invalid, fitnesses):
        ind.fitness.values = fit
    return population

if __name__ == "__main__":
    pop_size = 50
    meta_features = ["n", "p"]
    pset = make_pset(meta_features)
    problem_dimension = 2
    mf = {"n":10, "p":5}

    # Koza-style tree depth limits.
    limit = static_limit(key=operator.attrgetter("height"), max_value=12)
    # Mutations + Adaption to N-D Problems
    mate = limit(deap.gp.cxOnePoint)
    mate =   partial(nd_crossover, cx1d=mate)
    mutate = limit(partial(deap.gp.mutUniform, expr=partial(deap.gp.genFull, min_=0, max_=2), pset=Individual.pset))
    mutate = partial(nd_mutation, mut1d=mutate)
    select = partial(deap.tools.selDoubleTournament, fitness_size=10, parsimony_size=1.6, fitness_first=True)
    NDIndividual.create_population = partial(NDIndividual.create_population, ndim = problem_dimension)

    algorithm = gp.algorithms.AgeFitness(
        mate, mutate, select, NDIndividual.create_population
    )
    pop = update_fitness(NDIndividual.create_population(pop_size), mfargs=mf)

    for gen in range(50):
        pop = algorithm.evolve(pop)
        pop = update_fitness(pop, mfargs=mf)
        best = deap.tools.selBest(pop, 1)[0]

        print(best, best.popt, best.fitness.values)

        if best.fitness.values[0] <= 1e-8:
            break






