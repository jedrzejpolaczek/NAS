#!/usr/bin/env python
"""
Genetic algorithm class
"""

import random
from loguru import logger


class GeneticAlgorithm(object):
    """Evolve a population iteratively to find better
    individuals on each generation. If elitism is set, the
    fittest individual of a generation will be part of the
    next one.
    """

    def __init__(self, population, tournament_size=5, elitism=True):
        self.population = population
        self.x_train, self.y_train = self.population.get_data()
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.generation = 1

    def get_population_type(self):
        return self.population.__class__

    def run(self, max_generations: int) -> None:
        """
        Execute the main genetic algorithm loop established a number of times.
        The main genetic algorithm loop contains evolving population method.

        :param max_generations (int): value to set how many times the main genetic algorithm loop need to be done.
        """
        logger.info("Starting genetic algorithm.")
        while self.generation <= max_generations:
            self.evolve_population()
            self.generation += 1

    def evolve_population(self):
        """Evolve current population using tournament, crossover and mutation."""
        if self.population.get_size() < self.tournament_size:
            raise ValueError("Population size is smaller than tournament size.")
        print("Evaluating generation #{}...".format(self.generation))
        fittest = self.population.get_fittest()
        print("Fittest individual is:")
        print(fittest)
        print("Fitness value is: {}\n".format(round(fittest.get_fitness(), 4)))
        new_population = self.get_population_type()(
            self.population.get_species(), self.x_train, self.y_train, individual_list=[],
            maximize=self.population.get_fitness_criteria()
        )
        if self.elitism:
            new_population.add_individual(self.population.get_fittest())
        while new_population.get_size() < self.population.get_size():
            child = self.tournament_select().reproduce(self.tournament_select())
            child.mutate()
            new_population.add_individual(child)
        self.population = new_population
        self.guard("evolve_population", "new_population", new_population)

    def tournament_select(self):
        """
        Choose fittest individual from random sub set of initial population.
        
        :retrun self.population.speciec.__class__: fittest individual from tournmanet.
        """
        tournament = self.get_population_type()(
            self.population.get_species(), 
            self.x_train, 
            self.y_train, 
            individual_list=[
                self.population[i] for i in random.sample(range(self.population.get_size()), self.tournament_size)
            ], 
            maximize=self.population.get_fitness_criteria()
        )
        
        fittest_individual = tournament.get_fittest()

        self.guard("tournament_select", "fittest_individual", fittest_individual)
        return tournament.get_fittest()

    @staticmethod
    def guard(fun_name: str, name: str, main_object, object_to_compare=None) -> None:
        """
        Guard method to check if data are correct and to log them.

        :param fun_name (str): from where we get the values.
        :param name (str): name of the value.
        :param main_object (object): object to be loged and maybe checked.
        :param object_to_compare (object): object to be compared with main object.
        """
        if object_to_compare is not None:
            assert main_object == object_to_compare
        logger.debug("{}:{} (type: {}): {}".format(fun_name, name, type(main_object), main_object))
