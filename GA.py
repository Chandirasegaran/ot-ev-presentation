import random
import numpy as np
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools

# Define the fitness function to be optimized (example: sphere function)
def fitness_function(individual):
    return sum(np.square(individual)),

# GA parameters
population_size = 100
chromosome_length = 10
mutation_rate = 0.01
max_generations = 100

# Create the fitness maximization problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create the toolbox
toolbox = base.Toolbox()
toolbox.register("attribute", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=chromosome_length)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", fitness_function)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=mutation_rate)
toolbox.register("select", tools.selTournament, tournsize=2)

# Initialize population
population = toolbox.population(n=population_size)

# Create an array to store the best fitness value of each generation
best_fitness = []

# Genetic algorithm
for generation in range(max_generations):
    # Evaluate fitness for each individual in the population
    fitness_values = [toolbox.evaluate(individual) for individual in population]
    
    # Find the best individual and its fitness value
    best_index = np.argmax(fitness_values)
    best_fitness.append(fitness_values[best_index])
    best_individual = population[best_index]
    
    # Display the best fitness value of the current generation
    print(f"Generation: {generation+1}, Best Fitness: {fitness_values[best_index]}")
    
    # Plot the fitness progress
    plt.plot(range(generation+1), best_fitness, 'b-')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Genetic Algorithm')
    plt.show(block=False)
    plt.pause(0.001)
    
    # Selection
    selected_individuals = toolbox.select(population, len(population))
    
    # Create the next generation
    offspring = [toolbox.clone(individual) for individual in selected_individuals]
    
    # Apply crossover and mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        toolbox.mate(child1, child2)
        toolbox.mutate(child1)
        toolbox.mutate(child2)
        del child1.fitness.values
        del child2.fitness.values
    
    # Update the population
    population[:] = offspring

# Plot the final best fitness progress
plt.plot(range(max_generations), best_fitness, 'b-')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Genetic Algorithm')
plt.show()
