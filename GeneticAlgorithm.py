import random
from array import array

import numpy as np
import sys

# Coordenadas x e y das cidades (substitua pelos dados reais)
x = [...]  # Defina as coordenadas x das cidades (20 elementos)
y = [...]  # Defina as coordenadas y das cidades (20 elementos)
best_cost = sys.maxsize
best_solution = None

def fitness(pop):
    # pop é um array 2D onde cada linha representa um cromossomo (caminho)
    Npop, Ncidades = pop.shape

    # Gera a matriz de rota onde cada rota retorna à cidade inicial
    tour = np.hstack((pop, pop[:, [0]]))

    # Calcula a distância entre as cidades
    dcidade = np.zeros((Ncidades, Ncidades))
    for i in range(Ncidades):
        for j in range(Ncidades):
            dcidade[i, j] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)

    # Calcula o custo de cada cromossomo - a soma das distâncias para cada indivíduo
    dist = np.zeros(Npop)
    for i in range(Npop):
        for j in range(Ncidades):
            dist[i] += dcidade[tour[i, j], tour[i, j + 1]]
    return dist

def get_ten_best_of_generation(dist_array):
    best_of_generation = []

    for i in range(10):
        distance = sys.maxsize
        position_of_best = -1
        for j in range(1, len(dist_array)): # Starts in 1 because the start never change
            if dist_array[j] < distance and j not in best_of_generation:
                distance = dist_array[j]
                position_of_best = j
        best_of_generation.append(position_of_best) # Appends the best in order

    return best_of_generation

def cycle_crossing_over(parents):
    sons = []
    for i in range(0, len(parents), 2): # Adds 2 for each iteration
        position = random.randint(1, 19)
        parent1, parent2 = parents[i], parents[i + 1]
        while 1:
            parent1, parent2 = crossing_over(parent1.copy(), parent2.copy(), position)
            position = has_duplicate(parent1, position)
            if position is None:
                sons.append(parent1)
                sons.append(parent2)
                break
    return sons

def crossing_over(parent1, parent2, position):
    chromosome1 = parent1[position]
    chromosome2 = parent2[position]
    parent1[position] = chromosome2
    parent2[position] = chromosome1
    return parent1, parent2

def has_duplicate(parent, initial_position):
    seen = set()  # Set because lookup time is O(1)
    n = len(parent)

    # Start from the initial position and loop over the array
    for i in range(initial_position, initial_position + n):
        current_position = i % n  # Wrap around using modulo
        if parent[current_position] in seen:
            return current_position
        seen.add(parent[current_position])

    return None


def check_best_solution(population, best_of_generation, dist_array):
    global best_solution
    global best_cost
    min_value = dist_array[best_of_generation[0]]
    if min_value < best_cost:
        best_solution = population[best_of_generation[0]]
        best_cost = min_value


def genetic_algorithm(population):
    dist_array = fitness(population)
    best_of_generation = get_ten_best_of_generation(dist_array)
    check_best_solution(population, best_of_generation, dist_array)
    best_parents = np.array([population[i] for i in best_of_generation])
    cycle_crossing_over(best_parents)
    new_generation = np.vstack((best_parents, cycle_crossing_over(best_parents)))
    return new_generation

if __name__ == "__main__":
    # Carregar dados de cidades
    data = np.loadtxt('cidades.mat')
    x, y = data[0], data[1]

    # Parâmetros do algoritmo genético
    total_cidade = 20
    total_populacao = 20

    # Geração inicial da população
    population = np.array([np.insert(np.random.permutation(np.arange(1, total_populacao)), 0, 0) for _ in range(total_cidade)])
    print("Size of population: " + str(total_populacao))
    print("Initial Population: ")
    print(population + 1)  # Add 1 when displaying
    for i in range(10000):
        population = genetic_algorithm(population)
    print("Final Population: ")
    print(population + 1)  # Add 1 when displaying
    print("Number of cities: " + str(len(x)))
    print("Best cost: " + str(best_cost))
    print("Best solution: " + str(best_solution + 1))  # Add 1 when displaying