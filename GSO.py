import numpy as np
import time


# Garter Snake Optimization Algorithm
def GSO(population, fobj, LB, UB, max_iterations):
    population_size, dimension = population.shape
    lb = LB[0, :]
    ub = UB[0, :]
    best_position = population[np.argmin([fobj(x) for x in population])]
    best_fitness = fobj(best_position)
    convergence = np.zeros((1, max_iterations))
    ct = time.time()
    for iteration in range(max_iterations):
        for i in range(population_size):
            candidate_position = population[i]
            candidate_fitness = fobj(candidate_position)

            if candidate_fitness < best_fitness:
                best_position = candidate_position
                best_fitness = candidate_fitness

            # Update the position of the current snake
            r = np.random.uniform(0, 1, dimension)
            population[i] = population[i] + r * (best_position - population[i])

            # Ensure the position is within bounds
            population[i] = np.maximum(population[i], lb)
            population[i] = np.minimum(population[i], ub)

        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")
        convergence[iteration] = best_fitness
    ct = time.time() - ct
    return best_fitness, best_position, ct
