#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runs an entire experiment for benchmarking PURE_RANDOM_SEARCH on a testbed.

CAPITALIZATION indicates code adaptations to be made.
This script as well as files bbobbenchmarks.py and fgeneric.py need to be
in the current working directory.

Under unix-like systems:
    nohup nice python exampleexperiment.py [data_path [dimensions [functions [instances]]]] > output.txt &

"""
import sys  # in case we want to control what to run via command line args
import time

import numpy as np

import bbobbenchmarks
import config_islands as cfg
import fgeneric
from genetic_functions import *

argv = sys.argv[1:]  # shortcut for input arguments

datapath = './dataisland' if len(argv) < 1 else argv[0]

dimensions = (2, 3, 5, 10, 20) if len(argv) < 2 else eval(argv[1])
function_ids = bbobbenchmarks.nfreeIDs if len(argv) < 3 else eval(argv[2])
# function_ids = bbobbenchmarks.noisyIDs if len(argv) < 3 else eval(argv[2])
instances = range(1, 6) + range(41, 51) if len(argv) < 4 else eval(argv[3])

opts = dict(algid='PUT ALGORITHM NAME',
            comments='PUT MORE DETAILED INFORMATION, PARAMETER SETTINGS ETC')
maxfunevals = '10 * dim'  # 10*dim is a short test-experiment taking a few minutes
# INCREMENT maxfunevals SUCCESSIVELY to larger value(s)
minfunevals = 'dim + 2'  # PUT MINIMAL sensible number of EVALUATIONS before to restart
maxrestarts = 100      # SET to zero if algorithm is entirely deterministic


def generate_initial_population(sizes, dim, min=0, max=1):
	islands = []
	for size in sizes:
		islands.append(min + np.random.rand(size, dim) * (max - min))
	return islands


def generate_new_population(population, fitness):
	offspring = []
	for i in population:
		if random.random() < cfg.RECOMB_PROB:
			selected_idx = selection(fitness, 2, cfg.TOURNAMENT_SIZE)
			crossed = crossover_2(
				population[selected_idx[0]], population[selected_idx[1]])
			offspring.append(mutation(crossed, cfg.MUTATATION_COEF))
		else:
			selected_idx = selection(fitness, 1, cfg.TOURNAMENT_SIZE)
			offspring.append(mutation(population[selected_idx[0]], cfg.MUTATATION_COEF))

	return offspring


# miration chooses x most fit and replaces 10 worst on destination island
def migrate(islands, fitness_function):
	for island_idx, island in enumerate(islands):
		island_fitness = get_fitness(island, fitness_function)
		best_idxs = np.argsort(island_fitness)[:cfg.MIGRATION_SIZE]

		# see topology and replace destination islands weakest with this current
		# islands strongest
		for neighbour_idx in cfg.TOPOLOGY[island_idx]:
			# print("Migrating from island[{}] to island[{}]".format(
				# island_idx, neighbour_idx))
			neighbour_island_fitness = get_fitness(
				islands[neighbour_idx], fitness_function)
			worst_idxs = np.argsort(neighbour_island_fitness)[
			                        -cfg.MIGRATION_SIZE:]  # take the worst from current island
			for worst_idx, best_idx in zip(worst_idxs, best_idxs):
				if island_fitness[best_idx] > neighbour_island_fitness[worst_idx]:
					islands[neighbour_idx][worst_idx] = islands[island_idx][best_idx]
	return islands


def run_optimizer(fun, dim, max_iters, ftarget=np.inf):
    islands = generate_initial_population(
    	cfg.POPULATION_SIZES, dim, -4, 4)
    ret = run_island_ae(fun, islands, max_iters, ftarget)
    return ret


def get_best_individual(islands, best_x_idx):
    i = 0
    for island in islands:
        for individual in island:
            if i == best_x_idx:
                return individual
            i += 1


def run_island_ae(fun, x_start, max_iters, ftarget):
    fitness_function = fun
    islands = x_start

    best_f = np.inf
    best_x = np.inf
    epoch = 0
    while epoch < max_iters:
        for i, population in enumerate(islands):
            fitness = get_fitness(population, fitness_function)
            population = generate_new_population(population, fitness)
            islands[i] = population
            fitness = get_fitness(population, fitness_function)
            # print("Island: {}; Epoch: {}; Point: {}; Min: {};".format(
            #     i, epoch, population[np.argmin(fitness)], np.min(fitness)))

        epoch += 1
        if (epoch%cfg.MIGRATION_INTERVAL) == 0:
            islands = migrate(islands, fitness_function)
            fitness = []
            for island in islands: 
                fitness.append(get_fitness(island, fitness_function))

            fitness_flat = np.hstack(fitness)
            best_f = fitness_flat.min()
            best_x_idx = fitness_flat.argmin()
            best_x = get_best_individual(islands, best_x_idx)

        if abs(ftarget - best_f) < cfg.EPSILON:
            return best_x

    return best_x
    

t0 = time.time()
np.random.seed(int(t0))

f = fgeneric.LoggingFunction(datapath, **opts)
for dim in dimensions:  # small dimensions first, for CPU reasons
    print("dimensions: ", dim)
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
