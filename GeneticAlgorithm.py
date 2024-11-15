import random
import numpy as np
import sys

# Coordinates x and y of cities (replace with actual data)
x = [...]  # Define the x coordinates of cities (20 elements)
y = [...]  # Define the y coordinates of cities (20 elements)
best_cost = sys.maxsize
best_solution = []


def fitness(pop):
    """
    Calculates the fitness of each individual in the population.

    The fitness is computed as the total distance of a tour starting and ending at the first city.

    :param pop: numpy.ndarray: 2D array where each row is an individual representing a tour.
    :return: numpy.ndarray: 1D array containing the fitness (distance) of each individual.
    """
    Npop, Ncidades = pop.shape
    tour = np.hstack((pop, pop[:, [0]]))
    dcidade = np.zeros((Ncidades, Ncidades))
    for i in range(Ncidades):
        for j in range(Ncidades):
            dcidade[i, j] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
    dist = np.zeros(Npop)
    for i in range(Npop):
        for j in range(Ncidades):
            dist[i] += dcidade[tour[i, j], tour[i, j + 1]]
    return dist


def get_ten_best_of_generation(dist_array):
    """
    Identifies the ten best individuals from the generation based on fitness values.

    :param dist_array: numpy.ndarray: Array of fitness values for the population.
    :return: list[int]: Indices of the ten best individuals.
    """
    best_of_generation = []
    for i in range(10):
        distance = sys.maxsize
        position_of_best = -1
        for j in range(1, len(dist_array)):  # Starts in 1 because the start never changes
            if dist_array[j] < distance and j not in best_of_generation:
                distance = dist_array[j]
                position_of_best = j
        best_of_generation.append(position_of_best)
    return best_of_generation


def cycle_crossing_over(parents):
    """
    Performs cycle crossover on a set of parents to generate offspring.

    :param parents: list[numpy.ndarray]: List of parent individuals.
    :return: list[numpy.ndarray]: List of offspring individuals.
    """
    sons = []
    for i in range(0, len(parents), 2):  # Adds 2 for each iteration
        position = random.randint(1, 19)
        parent1, parent2 = parents[i], parents[i + 1]
        while True:
            parent1, parent2 = crossing_over(parent1.copy(), parent2.copy(), position)
            position = has_duplicate(parent1, position)
            if position is None:
                sons.append(parent1)
                sons.append(parent2)
                break
    return sons


def crossing_over(parent1, parent2, position):
    """
    Swaps genes at a specific position between two parents.

    :param parent1: numpy.ndarray: First parent.
    :param parent2: numpy.ndarray: Second parent.
    :param position: int: Position where the swap occurs.
    :return: tuple[numpy.ndarray, numpy.ndarray]: Updated parent1 and parent2.
    """
    chromosome1 = parent1[position]
    chromosome2 = parent2[position]
    parent1[position] = chromosome2
    parent2[position] = chromosome1
    return parent1, parent2


def has_duplicate(parent, initial_position):
    """
    Checks for duplicate genes in a parent starting from a given position.

    :param parent: numpy.ndarray: An individual's chromosome.
    :param initial_position: int: Starting position to check for duplicates.
    :return: int or None: Position of the duplicate gene, or None if no duplicates exist.
    """
    seen = set()
    n = len(parent)
    for i in range(initial_position, initial_position + n):
        current_position = i % n
        if parent[current_position] in seen:
            return current_position
        seen.add(parent[current_position])
    return None


def check_best_solution(population, best_of_generation, dist_array):
    """
    Updates the global best solution and cost if a better solution is found.

    :param population: numpy.ndarray: Current population of individuals.
    :param best_of_generation: list[int]: Indices of the best individuals in the generation.
    :param dist_array: numpy.ndarray: Fitness values for the population.
    """
    global best_solution
    global best_cost
    min_value = dist_array[best_of_generation[0]]
    if min_value < best_cost:
        best_solution = population[best_of_generation[0]]
        best_cost = min_value


def genetic_algorithm(population):
    """
    Performs one iteration of the genetic algorithm.

    :param population: numpy.ndarray: Current population of individuals.
    :return: numpy.ndarray: Updated population after applying selection and crossover.
    """
    dist_array = fitness(population)
    best_of_generation = get_ten_best_of_generation(dist_array)
    check_best_solution(population, best_of_generation, dist_array)
    best_parents = np.array([population[i] for i in best_of_generation])
    new_generation = np.vstack((best_parents, cycle_crossing_over(best_parents)))
    return new_generation


def creates_initial_generation(total_population, total_cities):
    """
    Creates the initial generation for a genetic algorithm.

    Each individual in the generation is a permutation of city indices,
    with the starting point (index 0) inserted at the beginning.

    :param total_population: int: Number of cities in each individual's route.
    :param total_cities: int: Number of individuals in the population.
    :return: numpy.ndarray: A 2D array where each row represents an individual's route.
    """
    return np.array([np.insert(np.random.permutation(np.arange(1, total_population)), 0, 0) for _ in range(total_cities)])

def log_data_beautify(array):
    """
    Add 1 to every value when displaying (values will go from 1 to 20)
    :param array: numpy.ndarray: Array of values.
    :return: numpy.ndarray: Array of log values.
    """
    return array + 1

if __name__ == "__main__":
    # Load city data
    data = np.loadtxt('cidades.mat')
    x, y = data[0], data[1]

    # Parameters of genetic algorithm
    total_cities = 20
    total_population = 20
    total_generations = 10000

    population = creates_initial_generation(total_population, total_cities)
    print("Size of population: " + str(total_population))
    print("Initial Population: ")
    print(log_data_beautify(population))  # Add 1 to every value when displaying (to go from 1 to 20)
    for i in range(total_generations):
        population = genetic_algorithm(population)
    print("Final Population: ")
    print(log_data_beautify(population))  # Add 1 when displaying
    print("Number of cities: " + str(len(x)))
    print("Best cost: " + str(best_cost))
    print("Best solution: " + str(log_data_beautify(best_solution)))  # Add 1 when displaying