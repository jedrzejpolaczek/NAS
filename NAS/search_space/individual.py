#!/usr/bin/env python
"""
Classes which define the individuals of a population with
its characteristic genes, generation, crossover and
mutation processes.
"""

import math
import pprint
import random
import numpy as np


def random_log_uniform(minimum, maximum, base, eps=1e-12):
    """Generate a random number which is uniform in a
    logarithmic scale. If base > 0 scale goes from minimum
    to maximum, if base < 0 vice versa, and if base is 0,
    use a uniform scale.
    """
    if base == 0:
        return random.uniform(minimum, maximum)
    minimum += eps  # Avoid math domain error when minimum is zero
    if base > 0:
        return base ** random.uniform(math.log(minimum, base), math.log(maximum, base))
    base = abs(base)
    return maximum - base ** random.uniform(math.log(eps, base), math.log(maximum - minimum, base))


class Individual(object):
    """
    Basic definition of an individual containing reproduction and mutation methods. 
    Do not instantiate, use a subclass which extends this object by defining a genome and a random individual generator.
    """

    def __init__(
        self, 
        x_train: np.array, 
        y_train: np.array, 
        genome:dict , 
        genes: str, 
        crossover_rate: float, 
        mutation_rate: float, 
        additional_parameters=None
    ):
        """
        Note:
        (GA) - means term is related to genetic algorithms.
        (NN) - means term is related to neural networks.

        :param x_train (numpy.ndarray): data input set (NN).
        :param y_train (numpy.ndarray): data output set (NN).
        :param genome (dict): stage with a number of nodes for it. Basing on that we will create neural network architecture. Genome is other name for individual.
        :param genes (str): string containing 0 and 1 to represent connections between nodes. Genes is the same thing as chromosome.
        :param crossover_rate (float): probability of crossover (GA). Crossover operation crossover two individuals. Default value is only example for easier class usage.
        :param mutation_rate (float): probability of mutation (GA). Mutation change random bits in individuals. Default value is only example for easier class usage.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.genome = genome
        self.validate_genome()
        self.genes = genes
        self.validate_genes()
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.fitness = None  # Until evaluated an individual fitness is unknown
        assert additional_parameters is None

    def validate_genome(self) -> None:
        """Check genome structure."""
        if type(self.genome) != dict:
            raise TypeError("Genome must be a dictionary.")

        for gene, properties in self.genome.items():
            if type(gene) != str:
                raise TypeError("Gene names must be strings.")

    def validate_genes(self) -> None:
        """Check that genes are compatible with genome."""
        if set(self.genome.keys()) != set(self.genes.keys()):
            raise ValueError("Genes passed don't correspond to individual's genome.")

    def get_genes(self) -> str:
        """
        Return individual's genes.

        :return str: individual's genes.
        """
        return self.genes

    def get_genome(self) -> dict:
        """
        Return individual's genome.
        
        :return dict: individual's genome.
        """
        return self.genome

    @staticmethod
    def generate_random_genes(genome) -> None:
        """
        Template method for generating random genes.
        
        :param genom (dict): dictionary of stages with a number of nodes for it.
        """
        raise NotImplementedError("Use a subclass with genes definition.")

    def evaluate_fitness(self) -> None:
        """Template method for evaluate fitness."""
        raise NotImplementedError("Use a subclass with genes definition.")

    def get_additional_parameters(self) -> None:
        """Template getter for additional parameters."""
        raise NotImplementedError("Use a subclass with genes definition.")

    def get_fitness(self) -> list:
        """
        Compute individual's fitness if necessary and return it.
        
        :return list: value of fitness in list. Type of list is due to fitness value can contain many values (like acc or val).
        """
        if self.fitness is None:
            self.evaluate_fitness()
        return self.fitness

    def reproduce(self, partner) -> None:
        """
        Mix genes from self and partner randomly and
        return a new instance of an individual. Do not mutate parents.

        :param partner (self.__class__): another object of this class to perform reproduce with.
        """
        assert self.__class__ == partner.__class__  # Can only reproduce if they're the same species
        child_genes = {}
        for name, value in self.get_genes().items():
            if random.random() < self.crossover_rate:
                child_genes[name] = partner.get_genes()[name]
            else:
                child_genes[name] = value
        
        # Create new same class individual
        return self.__class__(
            self.x_train, self.y_train, self.genome, child_genes, self.crossover_rate, self.mutation_rate,
            **self.get_additional_parameters()
        )

    def crossover(self, partner) -> None:
        """
        Mix genes from self and partner randomly.
        Mutates each parent instead of producing a new instance (child).

        :param partner (self.__class__): another object of this class to perform crossover with.
        """
        assert self.__class__ == partner.__class__  # Can only cross if they're the same species
        for name in self.get_genes().keys():
            if random.random() < self.crossover_rate:
                self.get_genes()[name], partner.get_genes()[name] = partner.get_genes()[name], self.get_genes()[name]
                self.set_fitness(None)
                partner.set_fitness(None)

    def mutate(self) -> None:
        """Mutate instance's genes with a certain probability."""
        for name, value in self.get_genes().items():
            if random.random() < self.mutation_rate:
                default, minimum, maximum, log_scale = self.get_genome()[name]
                if type(default) == int:
                    self.get_genes()[name] = random.randint(minimum, maximum)
                else:
                    self.get_genes()[name] = round(random_log_uniform(minimum, maximum, log_scale), 4)
                self.set_fitness(None)  # The mutation produces a new individual

    def get_fitness_status(self) -> bool:
        """
        Return True if individual's fitness in known.

        :return bool: Return True if individual's fitness in known.
        """
        return self.fitness is not None

    def set_fitness(self, value: list) -> None:
        """
        Assign fitness.
        
        :param value (list): value to be assign to fitness.
        """
        self.fitness = value

    def copy(self):
        """
        Copy instance of individual.
        
        :return self.__class__: copy of the individual instance.
        """
        individual_copy = self.__class__(
            self.x_train, self.y_train, self.genome, self.genes.copy(), self.crossover_rate,
            self.mutation_rate, **self.get_additional_parameters()
        )
        individual_copy.set_fitness(self.fitness)

        return individual_copy

    def __str__(self) -> str:
        """
        Return genes which identify the individual.

        :return str: genes which identify the individual."""
        return pprint.pformat(self.genes)
