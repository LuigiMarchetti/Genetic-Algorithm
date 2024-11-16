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
    tour = np.hstack((pop, pop[:, [0]]))  # Add the first city to the end of the tour
    dcidade = np.zeros((Ncidades, Ncidades))  # Distance matrix
    for i in range(Ncidades):
        for j in range(Ncidades):
            dcidade[i, j] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)  # Euclidean distance between cities

    dist = np.zeros(Npop)  # Array to store distances for each individual
    for i in range(Npop):
        for j in range(Ncidades):
            dist[i] += dcidade[tour[i, j], tour[i, j + 1]]  # Calculate total distance for each individual
    return dist


def get_population_ordered(dist_array):
    """
    Identifies the ten best individuals from the generation based on fitness values.

    :param dist_array: numpy.ndarray: Array of fitness values for the population.
    :return: list[int]: Indices of the ten best individuals.
    """
    best_of_generation = []
    for i in range(20):
        distance = sys.maxsize
        position_of_best = -1
        for j in range(1, len(dist_array)):  # Starts at 1 to avoid the first city as it's fixed
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
    for i in range(0, len(parents), 2):  # Process pairs of parents
        position = random.randint(1, 19)
        parent1, parent2 = parents[i], parents[i + 1]
        while True:
            parent1, parent2 = crossing_over(parent1.copy(), parent2.copy(), position)  # Perform crossover
            position = has_duplicate(parent1, position)  # Check if duplicate exists after crossover
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
            return current_position  # Return position of the duplicate
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


def get_new_generation_parents(population, dist_array):
    """
    Selects new parents using a roulette wheel selection based on fitness values.

    :param population: numpy.ndarray: Current population of individuals.
    :param dist_array: numpy.ndarray: Fitness values for the population.
    :return: list[numpy.ndarray]: List of selected parents.
    """
    total_fitness = sum(dist_array)
    roulette = [0]  # Initialize roulette wheel with zero
    for i in range(len(population)):
        probability_of_parent = dist_array[i] / total_fitness
        roulette.append(roulette[i] + probability_of_parent)

    parents = []
    number_of_parents = 10
    for _ in range(number_of_parents):
        number = random.random()  # Random number between 0 and 1
        for i in range(len(roulette) - 1):
            if roulette[i] < number < roulette[i + 1]:
                parents.append(population[i])

    return parents


def select_half_the_population(population, generation_ordered):
    """
    Selects the top half of the population based on fitness values.

    :param population: numpy.ndarray: Current population of individuals.
    :param generation_ordered: list[int]: Indices of the best individuals in the generation.
    :return: list[numpy.ndarray]: Selected individuals.
    """
    best_of_last_generation = generation_ordered[:10]
    best_half = [population[i] for i in best_of_last_generation]
    return best_half


def mutate_generation(new_generation):
    """
    Performs mutation by swapping two cities in each individual.

    :param new_generation: list[numpy.ndarray]: List of individuals to mutate.
    :return: list[numpy.ndarray]: List of mutated individuals.
    """
    for i in range(len(new_generation)):
        position1 = random.randint(1, 19)
        position2 = random.randint(1, 19)
        new_generation[i][position1], new_generation[i][position2] = new_generation[i][position2], new_generation[i][position1]
    return new_generation


def genetic_algorithm(population):
    """
    Performs one iteration of the genetic algorithm.

    :param population: numpy.ndarray: Current population of individuals.
    :return: numpy.ndarray: Updated population after applying selection, crossover, and mutation.
    """
    dist_array = fitness(population)  # Calculate fitness values for the population
    generation_ordered = np.array(get_population_ordered(dist_array))  # Get the order of best individuals
    check_best_solution(population, generation_ordered, dist_array)  # Updates best solution if necessary
    parents = get_new_generation_parents(population, dist_array)  # Select parents for crossover

    best_of_last_generation = select_half_the_population(population, generation_ordered)  # Select top half of the population
    new_child = cycle_crossing_over(parents)  # Perform crossover to generate offspring
    new_generation = np.vstack((best_of_last_generation, new_child))  # Combines best individuals with new offspring

    new_generation = mutate_generation(new_generation)  # Perform mutation on the new generation

    return np.array(new_generation)


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
    Adjusts the display format by adding 1 to each value to shift from 0-based to 1-based indexing.

    :param array: numpy.ndarray: Array of values to be displayed.
    :return: numpy.ndarray: Adjusted array for display.
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
    for _ in range(total_generations):
        population = genetic_algorithm(population)
    print("Final Population: ")
    print(log_data_beautify(population))  # Add 1 when displaying
    print("Number of cities: " + str(len(x)))
    print("Best cost: " + str(best_cost))
    print("Best solution: " + str(log_data_beautify(best_solution)))  # Add 1 when displaying
