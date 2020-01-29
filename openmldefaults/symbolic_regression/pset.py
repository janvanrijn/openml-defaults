import operator
import math
import random
import numpy as np
from deap import gp

def register_pset():
    np.seterr(all='raise')

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

    def log_abs(x):
        if (x < 0):
            x = abs(x)
        try:
            if (x == 0):
                raise ValueError
            return(np.log(x))
        except:
            return(1)

    # Operations
    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2, name = "div")
    pset.addPrimitive(operator.neg, 1)
    # sqrt and log are only def'd for x > 0
    pset.addPrimitive(sqrt_abs, 1, name = "sqrt")
    pset.addPrimitive(log_abs, 1, name = "log")
    pset.addPrimitive(protectedPow, 2, name = "pow")
    pset.addEphemeralConstant("rand01", lambda: random.uniform(0,1))
    pset.addEphemeralConstant("int010", lambda: random.randint(1, 10))
    pset.addEphemeralConstant("pow10", lambda: protectedPow(10, random.randint(-3, 3)))
    pset.renameArguments(ARG0='x')
    return(pset)
