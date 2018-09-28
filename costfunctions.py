'''
Python class containing various cost functions for testing optimisation algorithms
'''

import numpy as np
from math import cos
from math import sin
from math import pi
from math import sqrt

class Costfuncs(object):

    def __init__(self):
        return

    # simple sum of squares function that can take an arbitrary n-dimensional vector x as its arg
    def sphere(self, x, ndim=None):
        if x is not None:
            return 1.0*(sum(x**2))
        else:
            return [0.0]*ndim

    # function with many local minima having global minimum at (0,0). Expects 2D list of args
    def coscos(self, x):
        if x is not None:
            return (-50.0*cos(x[0])*cos(x[1]) + x[0]**2 + x[1]**2)
        else:
            return (0.0, 0.0)

    # Himmelblau's function is a multi-modal function with 4 identical local minima and one local maximum.
    # Range: -5 <= x,y <= +5. Expects 2D list of args
    def himmelblau(self, x, maximise=False):
        if x is not None:
            return ((x[0]**2 + x[1] - 11.0)**2 + (x[0] + x[1]**2 - 7.0)**2)
        if x is None and maximise:
            return (-0.270845, -0.923039)

    # Rastrigin function. Has search space (-5.12, 5.12) for all x_i and large no. of local minima. Expects n-dimensional list of args
    def rastrigin(self, x, ndim=None):
        if x is not None:
            return (10.0*len(x) + sum([i**2 - 10.0*cos(2.0*pi*i) for i in x]))
        else:
            return [0.0]*ndim

    # Griewank function. Expects n-dimensional list of args. Has a v large no, of widespread,
    # regularly distributed local minima. Search domain (-600, 600) for all x_i
    # global min. f(x) = 0 at all x_i = 0.
    def griewank(self, x, ndim=2):
        dim = np.arange(1,ndim+1)
        if x is not None:
            return np.sum(np.square(x)/4000.0) - (np.prod(x/np.sqrt(dim))) + 1.0
        else:
            return [0.0]*ndim

    # Three-hump camel function. Expects 2D list of args. Search domain (-5, 5) for each arg.
    # has 3 local minima, incl. global min at (0., 0.)
    def threehump_camel(self, x):
        if x is not None:
            return (2.0*x[0]**2) - (1.05*x[0]**4) + (x[0]**6/6.0) + (x[0]*x[1]) + (x[1]**2)
        else:
            return (0.0, 0.0)

    # Six-hump camel function. Expects 2D list of args. Search domain ((-3.0, 3.0), (-2.0, 2.0)
    # Has six local minima incl two global minima
    def sixhump_camel(self, x):
        if x is not None:
            return (4.0 - (2.1*np.square(x[0])) + ((x[1]**4)/3.0))*np.square(x[0]) + (x[0]*x[1]) + \
                (-4.0 + (4.0*np.square(x[1])))*np.square(x[1])
        else:
            return ((0.0898, -0.7126), (-0.0898, 0.7126))


    # Schwefel function. Expects n-dimensional list of args. Search domain (-500, 500) for all x_i.
    # Has a very large number of local minima, and a global min. f(x) = 0 at all x_i = 420.9687
    def schwefel(self, x, ndim=2):
        if x is not None:
            return 418.9829*ndim - np.sum(x*sin(np.sqrt(np.absolute(x))))
        else:
            return [420.9687]*ndim

    # Rosenbrock's 'banana' function. Non-convex. Expects n-dimensional list of args. To find the flat valley is
    # trivial, to converge the global min is difficult. For 2 <= N <= 7, global min is all ones. For N=3, this is the only min
    def rosenbrock(self, x, ndim=2):
        if x is not None:
            val = 0.
            for i in range(ndim-1):
                val += 100.*(x[i+1] - x[i]**2)**2 + (1.-x[i]**2)**2
            return val
        else:
            return [1]*ndim

    # Eggbox function. Many minima but strong funnelling to global minimum at the origin. Usual test range is [-5,5] for all i.
    # Expects 2D list of args
    def eggbox(self, x):
        if x is not None:
            return x[0]**2 + x[1]**2 + 25.*(sin(x[0])**2+sin(x[1])**2)
        else:
            return [0., 0.]

    # Ackley function. Takes N-dimensional list of arguments. Global minimum at the origin. Usual test domain is
    # [-5,+5] for all i. Has a very large number of local minima but overall topology strongly funnels to global minimum.
    def ackley(self, x, ndim=2):
        if x is not None:
            sum1, sum2 = 0., 0.
            for i in range(ndim):
                sum1 += x[i]**2
                sum2 += cos((2.*pi)*x[i])
            return 20. + np.exp(1) -20.*exp(-0.2*sqrt((1./ndim)*sum1)) - exp((1./ndim)*sum2)
        else:
            return [0.]*ndim
