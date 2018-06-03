import numpy as np
import random
from genetic_functions import *

# PROBLEM INSTANCES
instances = []
EPSILON = 0.0001
MAX_ITERS = 1000
POPULATION_SIZE = 100
TOURNAMENT_SIZE = int(POPULATION_SIZE / 10)


def generate_initial_population(size, dim, min=0, max=1):
	"""

	:param max:
	:param min:
	:param dim: dimensions of each element
	:param size: number of elements
	:return: random population
	"""
	return min + np.random.rand(size, dim) * (max - min)

def main():
	population = generate_initial_population(POPULATION_SIZE, 2, -10, 10)
	fitness_function = tmp_fun

	fitness = get_fitness(population, fitness_function)
	print(population)
	epoch = 0
	while epoch < MAX_ITERS:
		offspring = []
		for i in population:
			if random.random() < 0.2:
				selected_idx = selection(fitness, 2, TOURNAMENT_SIZE)
				# print(selected_idx)
				crossed = crossover_2(population[selected_idx[0]], population[selected_idx[1]])
				offspring.append(mutation(crossed))
			else:
				selected_idx = selection(fitness, 1, TOURNAMENT_SIZE)
				offspring.append(mutation(population[selected_idx[0]]))

		population = offspring
		epoch = epoch + 1

		fitness = get_fitness(population, fitness_function)
		print("Epoch: {}; Point: {}; Min: {};".format(epoch, population[np.argmin(fitness)], np.min(fitness)))


if __name__ == '__main__':
	main()
