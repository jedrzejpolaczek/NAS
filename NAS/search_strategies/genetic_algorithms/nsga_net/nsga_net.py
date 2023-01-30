#!/usr/bin/env python
"""
NSGANet class.
"""

from loguru import logger
import keras_tuner

try:
    from ..nsga_2.nsga_2 import NSGA2
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
        best_model, best_hyperparameters = self.bayesian_optimization_algorithm()

        logger.info("Best best hyperparameters: {}".format(best_hyperparameters))

    def bayesian_optimization_algorithm(self, epochs=30):  # TODO: add typing and docstring
        """
        Sampling from the Bayesian Network (BN) constructed by NSGA-Net.

        :return tensorflow.keras.models.Model: model with best hyperparameters.
        """
        # TODO: add exploitation phase - BOA -> https://arxiv.org/pdf/1810.03522.pdf -> 3.2
        best_model = self.population.get_fittest().get_model()
        train_data_set = (self.population.x_train, self.population.y_train)
        test_data_set = (self.population.x_test, self.population.y_test)

        tuner = keras_tuner.tuners.BayesianOptimization(
                best_model,
                objective='val_accuracy',
                max_trials=50
        )

        tuner.search(
                train_data_set,
                validation_data=test_data_set,
                epochs=epochs
        )

        best_model = tuner.get_best_models(1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

        return best_model, best_hyperparameters
