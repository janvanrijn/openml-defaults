from deap import tools
from sympy import simplify
import numpy as np

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

def draw_graph(ind, path):
    import pygraphviz as pgv
    nodes, edges, labels = gp.graph(ind)
    g = pgv.AGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    g.layout(prog="dot")

    for i in nodes:
        n = g.get_node(i)
        n.attr["label"] = labels[i]

    g.draw(path)
