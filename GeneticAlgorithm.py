import numpy as np

# Global variables (x and y coordinates of cities)
x = [...]  # Define the x-coordinates of the cities
y = [...]  # Define the y-coordinates of the cities

def cvfun(pop):
    # pop is a 2D array where each row represents a chromosome (tour/path)
    Npop, Ncidades = pop.shape

    # Generate the tour matrix where each tour returns to the starting city
    tour = np.hstack((pop, pop[:, [0]]))

    # Calculate the distance between cities
    dcidade = np.zeros((Ncidades, Ncidades))
    for i in range(Ncidades):
        for j in range(Ncidades):
            dcidade[i, j] = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)

    # Calculate the cost of each chromosome - the sum of distances for each individual
    dist = np.zeros(Npop)
    for i in range(Npop):
        for j in range(Ncidades):
            dist[i] += dcidade[tour[i, j], tour[i, j + 1]]

    return dist

if __name__ == "__main__":
    data = np.loadtxt('cidades.mat')
    x, y = data[0], data[1]

    print(x)
    print(y)

