import numpy as np
import math

def read_input(file):
    with open (file, "r") as file:
        num = int(file.readline().strip())
        locations = []

        for ind in range(num):
            line = file.readline().strip().split()
            location = (int(line[0]),int(line[1]), int(line[2]))
            locations.append(location)
    return locations

def DistanceMatrix(locations):
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(i + 1, num_locations):   
            dist = np.linalg.norm(np.array(locations[i]) - np.array(locations[j]))
            distance_matrix[i][j] = dist
            distance_matrix[j][i] = dist   

    return distance_matrix

def DistancePath(path, distance_matrix):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i]][path[i + 1]]
    total_distance += distance_matrix[path[-1]][path[0]]
    return total_distance


def CreateInitialPopulation(size, locations):
    num_locations = len(locations)  
    initial_population = []
    for ind in range(size):
        initial_population.append(np.random.permutation(num_locations).tolist())
    return initial_population


def CalculateRankList(paths, distance_matrix):
    fitness_scores = [1 / DistancePath(path, distance_matrix) for path in paths]
    rank_list = sorted(enumerate(fitness_scores), key=lambda x: x[1], reverse=True)
    return rank_list

def CreateMatingPool(population, RankList):
    fitness_scores = [fitness for index, fitness in RankList]
    sum_fitness = 0
    
    if sum_fitness == 0:
        return population[RankList[0][0]]
    
    for score in (fitness_scores):
        sum_fitness += score
    rand_var = np.random.uniform(0,sum_fitness)
    total = 0
    
    for ind, fit in RankList:
        total += fit
        if total > rand_var:
            return population[ind]


def CrossOverParents(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    start = np.random.randint(0, size - 2)
    end = np.random.randint(start + 1, size - 1)

    child[start:end+1] = parent1[start:end+1]
    for i in range(start, end + 1):
        if parent2[i] not in child:
            ind = i
            while child[ind] is not None:
                ind = parent2.index(parent1[ind])
            child[ind] = parent2[i]
    for i in range(size):
        if child[i] is None:
            child[i] = parent2[i]

    return child

def ChooseBestIndividuals(population, RankListForPopulation):
    size = len(population)

    num_chosen = int(size*0.2)

    if size > 2 and num_chosen < 2:
        num_chosen = 2

    a= np.ceil(11);
    bestIndividuals = []
    for i in range(num_chosen):
        ind, fitness = RankListForPopulation[i]
        bestIndividuals.append(population[ind])
    return bestIndividuals

 
def Mutate(path, mutation_rate):
    size = len(path)
    if np.random.rand() < mutation_rate:
        start = np.random.randint(0, size - 2)
        end = np.random.randint(start + 1, size - 1)
        path[start], path[end] = path[end], path[start]
    return path

def GATS(locations):
    size = len(locations)
    if size <= 150:
        population_size = 150
        iteration = 300
    else:        
        population_size=350
        iteration=800
    mutation_rate = 0.03
    if len(locations) == 0:
        return 0, []
    if len(locations) == 1:
        return 0, [0]
    if len(locations) == 2:
        return DistancePath([0, 1], DistanceMatrix(locations)), [0, 1]
    distance_matrix = DistanceMatrix(locations)  
    population = CreateInitialPopulation(population_size, locations)
    best_path = None
    best_distance = float("inf")

    repeat_count = 0
    repeat_limit = 130

    for generation in range(iteration):
        RankListForPopulation = CalculateRankList(population, distance_matrix)
        new_population = ChooseBestIndividuals(population, RankListForPopulation)
 
        while len(new_population) < population_size:
            parent1 = CreateMatingPool(population, RankListForPopulation)
            parent2 = CreateMatingPool(population, RankListForPopulation)

            child1 = CrossOverParents(parent1, parent2)
            child2 = CrossOverParents(parent1, parent2)
            
            child1 = Mutate(child1, mutation_rate)
            child2 = Mutate(child2, mutation_rate)
            
            new_population.extend([child1, child2])

        population = new_population

        current_best = population[RankListForPopulation[0][0]]
        current_best_distance = DistancePath(current_best, distance_matrix)
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_path = current_best
            repeat_count = 0
        else:
            repeat_count += 1

        if repeat_count > repeat_limit:
            break
    
    return best_distance, best_path

def write_output(total_distance, path, locations, file_name):
    with open(file_name, "w") as f:
        f.write(f"{total_distance:.3f}\n")
        for ind in path:
            loc = locations[ind]
            f.write(f"{loc[0]} {loc[1]} {loc[2]}\n")
        if path:    
            first_loc = locations[path[0]]
            f.write(f"{first_loc[0]} {first_loc[1]} {first_loc[2]}\n")

if __name__ == "__main__":

    initial_locations = read_input("input.txt")
    best_distance, best_path = GATS(initial_locations)
    write_output(best_distance, best_path, initial_locations, "output.txt")

