import random
import numpy as np

# tournament selection
def selection(fitness, k, tournament_size):
	won = []
	indices = range(len(fitness))
	for i in range(k):
		chosen = random.sample(indices, tournament_size)
		chosen_fit = []
		for x in chosen:
			chosen_fit.append(fitness[x])
		won.append(chosen[np.argmin(chosen_fit)])

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


def tmp_fun(individual):
	return np.sum([gene**2 for gene in individual])
	
