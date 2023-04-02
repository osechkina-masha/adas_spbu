from .individual import Individual
from ...environment import Environment
from typing import List, Tuple, Optional
from ...description import NormalizedParameters

import numpy as np
import random


Population = List[Individual]


def select(population: Population, target_size: int, env: Environment, tourn_size: int = 2) -> Tuple[Population, List[float]]:
    population_params = [env.parameters_description.decode_parameters(ind.behave()) for ind in population]

    fitness = [env.score(p) for p in population_params]
    fitness = np.array(fitness)

    new_population = []
    new_fitness = []

    old_population_size = len(population)
    for _ in range(target_size):
        sample_ind = [random.randint(0, old_population_size - 1) for _ in range(tourn_size)]
        best = sample_ind[np.argmax(fitness[sample_ind])]
        new_population.append(population[best])
        new_fitness.append(fitness[best])
    return new_population, new_fitness


def crossover(population: Population, cros_pb: float) -> Population:
    new_pop = []
    for ind1, ind2 in zip(population[::2], population[1::2]):
        if random.random() < cros_pb:
            new_pop.extend([
                ind1.crossover(ind2),
                ind2.crossover(ind1)
            ])
    return new_pop


def mutation(population: Population, mut_pb: float) -> Population:
    for ind in population:
        if random.random() < mut_pb:
            ind.mutate()
    return population


def ellitism(population: Population, fitness: List[float], n: int = 5) -> Tuple[Population, Population]:
    elite_ind = np.argpartition(np.array(fitness), -n)[-n:]
    elite = []
    other = []
    for i, ind in enumerate(population):
        if i in elite_ind:
            elite.append(ind)
        else:
            other.append(ind)
    return elite, other


def best_fitness(population: Population, env: Environment) -> Individual:
    parameters = [i.behave() for i in population]
    scaled_p = [env.parameters_description.decode_parameters(p) for p in parameters]
    fitness = [env.score(p) for p in scaled_p]
    return population[np.argmax(fitness)]


def run_genetic(env: Environment,
                n_generations: int,
                pop_size: int,
                tourn_size: int = 2,
                p_crossover: float = 0.9,
                p_mutation: float = 0.1,
                elite_size: Optional[int] = None) -> NormalizedParameters:
    if elite_size is None:
        elite_size = int(0.2 * pop_size)
    population = [Individual(env.parameters_description) for _ in range(pop_size)]
    for generation in range(n_generations):
        population, fitness = select(population, pop_size, env, tourn_size)
        elite, others = ellitism(population, fitness, n=elite_size)
        others = crossover(others, p_crossover)
        others = mutation(others, p_mutation)
        population = elite + others
    best = best_fitness(population, env)
    return best.behave()
