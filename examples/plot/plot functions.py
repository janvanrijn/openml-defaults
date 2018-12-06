import numpy as np
import math
import matplotlib.pyplot as plt



fns = {
    'linear': lambda a, x: a * x,
    'square root': lambda a, x: np.sqrt(x) * a,
    'logaritmic': lambda a, x: np.log2(x) * a,
    'inverse': lambda a, x: a / x,
    'exponential': lambda a, x: x ** a,
}

for fn_name, fn in fns.items():
    plt.clf()
    ax = plt.subplot(111)
    geom_space = np.geomspace(0.01, 2, 10)
    geom_space = np.append(geom_space, [1])

    for a in geom_space:
        x = np.arange(0.0, 5.0, 0.01)
        s = fn(a, x)

        line, = plt.plot(x, s, lw=2)

    plt.savefig(fn_name.replace(' ', '_') + '.eps')
