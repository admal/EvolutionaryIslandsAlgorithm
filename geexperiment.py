#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runs an entire experiment for benchmarking PURE_RANDOM_SEARCH on a testbed.

CAPITALIZATION indicates code adaptations to be made.
This script as well as files bbobbenchmarks.py and fgeneric.py need to be
in the current working directory.

Under unix-like systems: 
    nohup nice python exampleexperiment.py [data_path [dimensions [functions [instances]]]] > output.txt &

"""
import sys # in case we want to control what to run via command line args
import time
import numpy as np
import fgeneric
import bbobbenchmarks
from genetic_functions import *

argv = sys.argv[1:] # shortcut for input arguments

datapath = './data' if len(argv) < 1 else argv[0]

dimensions = (2, 3, 5, 10, 20) if len(argv) < 2 else eval(argv[1])
function_ids = bbobbenchmarks.nfreeIDs if len(argv) < 3 else eval(argv[2])  
# function_ids = bbobbenchmarks.noisyIDs if len(argv) < 3 else eval(argv[2])
instances = range(1, 6) + range(41, 51) if len(argv) < 4 else eval(argv[3])

opts = dict(algid='PUT ALGORITHM NAME',
            comments='PUT MORE DETAILED INFORMATION, PARAMETER SETTINGS ETC')
maxfunevals = '10 * dim' # 10*dim is a short test-experiment taking a few minutes 
# INCREMENT maxfunevals SUCCESSIVELY to larger value(s)
minfunevals = 'dim + 2'  # PUT MINIMAL sensible number of EVALUATIONS before to restart
maxrestarts = 100      # SET to zero if algorithm is entirely deterministic 
POPULATION_SIZE = 100
TOURNAMENT_SIZE = int(POPULATION_SIZE / 10)
EPSILON = 1e-04
MUTATION_COEF = 0.001
MUTATION_PROB = 0.02
RECOMBINATION_PROB = 0.02

def generate_initial_population(size, dim, min=0, max=1):
	"""
	:param max:
	:param min:
	:param dim: dimensions of each element
	:param size: number of elements
	:return: random population
	"""
	return min + np.random.rand(size, dim) * (max - min)

def run_optimizer(fun, dim, max_iters, ftarget=np.inf):
    population = generate_initial_population(POPULATION_SIZE, dim, -4, 4)
    ret = run_basic_ae(fun, population, max_iters, ftarget)
    return ret
 
def run_basic_ae(fun, x_start, max_iters, ftarget):
    fitness_function = fun
    population = x_start
    fitness = get_fitness(population, fitness_function)
    # print(population)
    best_f = np.inf
    best_x = np.inf
    epoch = 0
    while epoch < max_iters and abs(ftarget - best_f) > EPSILON :
        offspring = []
        for i in population:
            if random.random() < 0.2:
                selected_idx = selection(fitness, 2, TOURNAMENT_SIZE)
                # print(selected_idx)
                crossed = crossover_2(population[selected_idx[0]], population[selected_idx[1]])
                offspring.append(mutation(crossed, MUTATION_COEF))
            else:
                selected_idx = selection(fitness, 1, TOURNAMENT_SIZE)
                offspring.append(mutation(population[selected_idx[0]], MUTATION_COEF))
 
        population = offspring
        epoch = epoch + 1
 
        fitness = get_fitness(population, fitness_function)
        best_f = np.min(fitness)
        best_x = population[np.argmin(fitness)]
 
    return best_x

t0 = time.time()
np.random.seed(int(t0))

f = fgeneric.LoggingFunction(datapath, **opts)
for dim in dimensions:  # small dimensions first, for CPU reasons
    for fun_id in function_ids:
        for iinstance in instances:
            f.setfun(*bbobbenchmarks.instantiate(fun_id, iinstance=iinstance))

            # independent restarts until maxfunevals or ftarget is reached
            for restarts in xrange(maxrestarts + 1):
                if restarts > 0:
                    f.restart('independent restart')  # additional info
                run_optimizer(f.evalfun, dim,  eval(maxfunevals) - f.evaluations,
                              f.ftarget)
                if (f.fbest < f.ftarget
                    or f.evaluations + eval(minfunevals) > eval(maxfunevals)):
                    break

            f.finalizerun()

            print('  f%d in %d-D, instance %d: FEs=%d with %d restarts, '
                  'fbest-ftarget=%.4e, elapsed time [h]: %.2f'
                  % (fun_id, dim, iinstance, f.evaluations, restarts,
                     f.fbest - f.ftarget, (time.time()-t0)/60./60.))

        print '      date and time: %s' % (time.asctime())
    print '---- dimension %d-D done ----' % dim