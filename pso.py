'''
Python script to perform particle swarm optimisation (PSO) on a provided n-dimensional cost function
'''

import numpy as np
import matplotlib
from copy import deepcopy
from costfunctions import Costfuncs

class Pso(object):

    def __init__(self, domain, N, nruns, costf, omega=0.2, phi_p=0.2, phi_g=0.5, \
             maxstep = 5.0, thresh=1.0E-04, globalopt=None, ndim=2, maximise=False, dump=0, \
             randseed=19):
        np.random.seed(randseed)
        self.domain = domain
        self.maximise = maximise
        self.N = N # no. of particles in swarm
        self.nruns = nruns # max. no. of iterations
        self.costf = costf # choice of cost function
        self.omega = omega # vel update coeff, weight of current velocity
        self.phi_p = phi_p # vel update coeff, weight of diff between current particle position and current best
        self.phi_g = phi_g # vel update coeff, weight of diff between current swarm position and current best
        self.maxstep = maxstep # maximum for random number draw in velocity updates
        self.thresh = thresh # error threshold
        self.globalopt = globalopt # provide if known and want to calculate error
        self.ndim = ndim # no. of dimensions (must match no. of variables of cost function)
        self.dump = dump # interval for dumping plot & error info
        self.r, self.v = self.init_posvel() # particles current position and velocity
        self.p = deepcopy(self.r) # particles best known position
        self.g = np.zeros(shape=self.ndim)
        self.update_swarm_best(initialise=True) # swarm's best known position
        return

    def plot_soln(self):
        return

    def init_posvel(self):
        # vels and posns initialised with uniformly distributed random vectors
        r = np.random.uniform(self.domain[0], self.domain[1], size=self.ndim*self.N)
        velbound = abs(self.domain[0] - self.domain[1])
        v = np.random.uniform(-velbound, velbound, size=self.ndim*self.N)
        return r, v

    def calc_err(self):
        currparterr = 0.0 # error based on all current particle positions
        bestparterr = 0.0 # error based on all best particle positions
        try:
            bestswarmerr = sum(abs(self.globalopt - self.g))
            for j in range(self.N):
                err = sum(abs(self.globalopt - self.r[j*self.ndim:j*self.ndim+self.ndim]))/self.N
                currparterr += err
                err = sum(abs(self.globalopt - self.p[j*self.ndim:j*self.ndim+self.ndim]))/self.N
                bestparterr += err
        except TypeError:
            bestswarmerr = 0.0 # error based on single best particle position of swarm
        return currparterr, bestparterr, bestswarmerr

    def update_swarm_best(self, initialise=False):
        if initialise: currcost_swarm = float("inf")
        elif not self.maximise: currcost_swarm = self.costf(self.g)
        else: currcost_swarm = -1.0*self.costf(self.g)
        for j in range(self.N):
            pos_new = self.r[j*self.ndim:j*self.ndim+self.ndim]
            if not self.maximise: newcost = self.costf(pos_new)
            else: newcost = -1.0*self.costf(pos_new)
            if not initialise:
                pos_old = self.p[j*self.ndim:j*self.ndim+self.ndim]
                if not self.maximise: currcost_particle = self.costf(pos_old)
                else: currcost_particle = -1.0*self.costf(pos_old)
                if newcost < currcost_particle: # update particle best
                    self.p[j*self.ndim:j*self.ndim+self.ndim] = deepcopy(pos_new)
            if newcost < currcost_swarm: # update swarm best
                self.g = deepcopy(pos_new)
                currcost_swarm = newcost

    def pso_optimise(self):
        i = 0
        currparterr = float("inf")
        print "\nSTEP\tCURRENT PARTICLE ERR\tBEST PARTICLE ERR\tBEST SWARM ERR\n"
        while i < self.nruns and currparterr >= self.thresh:
            for j in range(self.N):
                for d in range(self.ndim):
                    w1 = np.random.uniform(0,self.maxstep)
                    w2 = np.random.uniform(0,self.maxstep)
                    # update velocities (displacements)
                    self.v[j*self.ndim+d] = self.omega*self.v[j*self.ndim+d] + \
                        self.phi_p*w1*(self.p[j*self.ndim+d] - self.r[j*self.ndim+d]) + \
                        self.phi_g*w2*(self.g[d] - self.r[j*self.ndim+d])
                # update positions
                self.r[j*self.ndim:j*self.ndim+self.ndim] = \
                    self.r[j*self.ndim:j*self.ndim+self.ndim] + \
                    self.v[j*self.ndim:j*self.ndim+self.ndim]
                continue
            self.r, self.p
            # update current bests
            self.update_swarm_best()
            if self.globalopt is not None:
                currparterr, bestparterr, bestswarmerr = self.calc_err()
                print i, "\t", currparterr, "\t", bestparterr, "\t", bestswarmerr
            i += 1
            if self.dump != 0 and i % self.dump == 0:
                self.plotsoln()
        return

### DRIVER CODE
if __name__ == "__main__":

    ### SET PARAMS
    N = 25 # number of particles in the swarm
    domain = (-3.0, 3.0) # (min, max) for all dimensions (e.g. both x & y coords if 2D)
    nruns = 100 # max. no. of runs
    costfunc1 = Costfuncs() # instance of Costfuncs class

    '''
    ### EXAMPLE 1
    globaloptcoords = costfunc1.himmelblau(x=None,maximise=True)
    costf = costfunc1.himmelblau
    costfuncdim = 2
    domain = (-1.5,0.)
    maxim = True
    '''
    '''
    ### EXAMPLE 2
    globaloptcoords = costfunc1.coscos(x=None) # n-dimensional vector giving coords at global optimimum.
    costf = costfunc1.coscos
    costfuncdim = 2
    maxim = False
    '''
    
    ### EXAMPLE 3
    globaloptcoords = costfunc1.rastrigin(x=None,ndim=3)
    costf = costfunc1.rastrigin
    N = 100 # This is a difficult problem so use more particles and more runs!
    nruns = 250
    costfuncdim = 3
    maxim = False
    
    '''
    ### EXAMPLE 4
    globaloptcoords = costfunc1.sphere(x=None,ndim=3)
    costf = costfunc1.sphere
    costfuncdim = 3
    maxim = False
    '''

    pso1 = Pso(domain,N,nruns,costf,ndim=costfuncdim,globalopt=globaloptcoords,maximise=maxim)
    pso1.pso_optimise()
