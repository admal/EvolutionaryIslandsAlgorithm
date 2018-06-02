import numpy as np
import random
import config_islands as cfg

def generate_initial_populations(sizes, dim, min=0, max=1):
	islands = []
	for size in sizes:
		islands.append(min + np.random.rand(size, dim) * (max - min))
	return islands


# tournament selection
def selection(fitness, k):
	won = []
	indices = range(len(fitness))
	for i in range(k):
		chosen = random.sample(indices, cfg.TOURNAMENT_SIZE)
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


def tmp_fun(individual):
	return np.sum([gene**2 for gene in individual])
	

def generate_new_population(population, fitness):
	offspring = []
	for i in population:
		if random.random() < cfg.RECOMB_PROB:
			selected_idx = selection(fitness, 2)
			crossed = crossover_2(population[selected_idx[0]], population[selected_idx[1]])
			offspring.append(mutation(crossed))
		else:
			selected_idx = selection(fitness, 1)
			offspring.append(mutation(population[selected_idx[0]]))

	return offspring


# miration chooses x most fit and replaces 10 worst on destination island
def migrate(islands, fitness_function):
	for island_idx, island in enumerate(islands):
		island_fitness = get_fitness(island, fitness_function)
		best_idxs = np.argsort(island_fitness)[:cfg.MIGRATION_SIZE]

		#see topology and replace destination islands weakest with this current islands strongest
		for neighbour_idx in cfg.TOPOLOGY[island_idx]:
			print("Migrating from island[{}] to island[{}]".format(island_idx, neighbour_idx))
			neighbour_island_fitness = get_fitness(islands[neighbour_idx], fitness_function)
			worst_idxs = np.argsort(neighbour_island_fitness)[-cfg.MIGRATION_SIZE:] # take the worst from current island
			for worst_idx, best_idx in zip(worst_idxs, best_idxs):
				if island_fitness[best_idx] > neighbour_island_fitness[worst_idx]:
					islands[neighbour_idx][worst_idx] = islands[island_idx][best_idx]
	return islands


def main():
	islands = generate_initial_populations(cfg.POPULATION_SIZES, cfg.GENOME_LEN, -10, 10)
	fitness_function = tmp_fun

	epoch = 0
	while epoch < cfg.MAX_ITERS:
		for i, population in enumerate(islands):
			fitness = get_fitness(population, fitness_function)
			population = generate_new_population(population, fitness)
			islands[i] = population
			fitness = get_fitness(population, fitness_function)
			print("Island: {}; Epoch: {}; Point: {}; Min: {};".format(i, epoch, population[np.argmin(fitness)], np.min(fitness)))
		
		epoch += 1
		if epoch%cfg.MIGRATION_INTERVAL == 0:
			islands = migrate(islands, fitness_function)
			fitness = []
			for island in islands:
				fitness.append(get_fitness(island, fitness_function))
			#print("global minimum across islands after migrations: ", np.array(fitness).flatten().min())	

	fitness = []
	for island in islands:
		fitness.append(get_fitness(island, fitness_function))
	print("global minimum: ", np.array(fitness).flatten().min())


if __name__ == '__main__':
	main()
