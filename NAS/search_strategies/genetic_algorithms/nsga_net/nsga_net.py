#!/usr/bin/env python
"""
NSGANet class.
"""

from loguru import logger

try:
    from ..nsga_2.nsga_2 import NSGA2
except ImportError:
    pass
try:
    from ..utils import bayesian_optimization_algorithm
except ImportError:
    pass


class NSGANet(NSGA2):
    """
    Class contain implementation of:
    Genetic algorithm used in the NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm papers by
    Zhichao Lu, Ian Whalen, Vishnu Boddeti, Yashesh Dhebar, Kalyanmoy Deb, Erik Goodman and Wolfgang Banzhaf from
    Michigan State University.

    Link to the papers: https://arxiv.org/pdf/1810.03522.pdf
    """

    def __init__(self, population, crossover_probability: float=0.2, mutation_probability: float=0.8):
        super(NSGANet, self).__init__(population)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
    
    def run(self, max_generations: int) -> None:
        """
        Execute the main genetic algorithm loop established a number of times.
        The main genetic algorithm loop contains evolving population method.
        At the end use Bayesian optimization algorithm on ricieved population.

        :param max_generations (int): value to set how many times the main genetic algorithm loop need to be done.
        """
        logger.info("Start exploration phase.")
        while self.generation <= max_generations:
            self.evolve_population()
            self.generation += 1

        logger.info("Start exploitation phase.")
        self.population = bayesian_optimization_algorithm(self.population)
