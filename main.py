import numpy as np
import random

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


# tournament selection
def selection(fitness, k):
	won = []
	indices = range(len(fitness))
	for i in range(k):
		chosen = random.sample(indices, TOURNAMENT_SIZE)
		won.append(np.argmin(fitness[chosen]))

	return won


def mutation(individual):
	new_individual = []
	for i in individual:
		r = -1 + random.random() * 2
		new_individual.append(i + r)
	return new_individual


def crossover(individuals):
	length = len(individuals[0])
	weights = np.random.rand(length)
	child = []
	for individual in individuals:
		for i in range(length):
			pass


def crossover_2(individual1, individual2):
	length = len(individual1)
	weights = np.random.rand(length)
	child = []
	for i in range(length):
		child.append(individual1[i] + weights[i] * (individual2[i] - individual1[i]))

	return child


def get_fitness(population, fitness_function):
	fitness = np.array([])
	for individual in population:
		fitness = np.append(fitness, fitness_function(individual))

	return fitness


def tmp_fun(x):
	return (x[0]) ** 2 + (x[1]) ** 2


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
				selected_idx = selection(fitness, 2)
				# print(selected_idx)
				crossed = crossover_2(population[selected_idx[0]], population[selected_idx[1]])
				offspring.append(mutation(crossed))
			else:
				selected_idx = selection(fitness, 1)
				offspring.append(mutation(population[selected_idx[0]]))

		population = offspring
		epoch = epoch + 1

		fitness = get_fitness(population, fitness_function)
		print("Epoch: {}; Point: {}; Min: {};".format(epoch, population[np.argmin(fitness)], np.min(fitness)))


if __name__ == '__main__':
	main()
