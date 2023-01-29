#!/usr/bin/env python
"""
Population class
"""

import itertools
import operator
from loguru import logger


class Population(object):
    """Group of individuals of the same species, that is,
    with the same genome. Can be initialized either with a
    list of individuals or a population size so that
    random individuals are created. The get_fittest method
    returns the strongest individual.
    """

    def __init__(self, 
        species, 
        x_train, 
        y_train, 
        individual_list: list=None, 
        size: int=None,
        crossover_rate: float=0.5, 
        mutation_rate: float=0.015, 
        maximize: bool=True,
        additional_parameters=None
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.species = species
        self.maximize = maximize
        if individual_list is None and size is None:
            raise ValueError("Either pass a list of individuals or a population size for a random population.")
        elif individual_list is None:
            if additional_parameters is None:
                additional_parameters = {}
            self.population_size = size
            self.individuals = [
                self.species(
                    self.x_train, self.y_train, crossover_rate=crossover_rate,
                    mutation_rate=mutation_rate, **additional_parameters
                )
                for _ in range(size)
            ]
            logger.debug("Initializing a random population. Size: {}".format(size))
        else:
            assert all([type(individual) is self.species for individual in individual_list])
            self.population_size = len(individual_list)
            self.individuals = individual_list

    def add_individual(self, individual):
        assert type(individual) is self.species
        self.individuals.append(individual)
        self.population_size += 1

    def get_species(self):  # In other (Python) words, object class
        return self.species

    def get_size(self):
        return self.population_size

    def get_fittest(self):
        if self.maximize:
            return max(self.individuals, key=operator.methodcaller('get_fitness'))
        return min(self.individuals, key=operator.methodcaller('get_fitness'))

    def get_data(self):
        return self.x_train, self.y_train

    def get_fitness_criteria(self):
        return self.maximize

    def __getitem__(self, item):
        return self.individuals[item]
