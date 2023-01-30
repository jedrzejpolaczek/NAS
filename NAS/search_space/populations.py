#!/usr/bin/env python
"""
Population class
"""

import numpy as np
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
            msg = "Either pass a list of individuals or a population size for a random population."
            logger.error(msg)
            raise ValueError(msg)
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

    def add_individual(self, individual) -> None:
        """
        Add individual to individual list (to population).

        :param individual (self.species.__class__): individual to ba added.
        """
        assert type(individual) is self.species
        self.individuals.append(individual)
        self.population_size += 1
        logger.debug("produce: {}".format(self.population_size))
        

    def get_species(self):  # In other (Python) words, object class
        """
        Return class of individuals in population.

        :return self.species.__class__: class of individuals in population.
        """
        logger.debug("return: {}".format(self.species))
        return self.species

    def get_size(self) -> int:
        """
        Return size of population.

        :return int: size of population..
        """
        logger.debug("return: {}".format(self.population_size))
        return self.population_size

    def get_fittest(self):
        """
        Return fittest individual from population.

        :return self.species.__class__: fittest individual from population.
        """
        if self.maximize:
            fittest = max(self.individuals, key=operator.methodcaller('get_fitness'))
        else:
            fittest = min(self.individuals, key=operator.methodcaller('get_fitness'))

        logger.debug("return: {}".format(fittest))
        return fittest

    def get_data(self):
        """
        Return data for neural network model to train on.

        :return numpy.array, numpy.array: data for neural network model to train on.
        """
        logger.debug("return: {}".format(self.x_train))
        logger.debug("return: {}".format(self.y_train))
        return self.x_train, self.y_train

    def get_fitness_criteria(self) -> bool:
        """
        Return fitness criteria.

        :return bool: fitness criteria.
        """
        logger.debug("return: {}".format(self.maximize))
        return self.maximize

    def __getitem__(self, item: int):
        """
        Return individual from individuals under pointed index.

        :param item (int): index pointed to specyfic individual in individuals list.

        :return self.species.__class__: individual from individuals under pointed index.
        """
        logger.debug("return: {}".format(self.individuals[item]))
        return self.individuals[item]
