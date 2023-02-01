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
        best_model = self.bayesian_optimization_algorithm()

        logger.info("Best model: {}".format(best_model))

    def bayesian_optimization_algorithm(self):  # TODO: add typing and docstring
        """
        Sampling from the Bayesian Network (BN) constructed by NSGA-Net.

        :return tensorflow.keras.models.Model: model with best hyperparameters.
        """
        # TODO: add exploitation phase - BOA -> https://arxiv.org/pdf/1810.03522.pdf -> 3.2
        from keras.optimizers import Adam
        def model_builder(hyperparameters):
            logger.debug("Load data.")
            logger.debug("hyperparameters: {}".format(hyperparameters))
            fittest_individual = self.population.get_fittest()
            fittest_individual_model_object = fittest_individual.get_model_class_object()
            hyperparameters_filters = hyperparameters.Int('filters', min_value = 32, max_value = 64, step = 32)

            logger.debug("Build the model.")
            model = fittest_individual_model_object.build_model(
                genes=fittest_individual.genes,
                nodes_per_stage=fittest_individual.nodes_per_stage, 
                input_shape=fittest_individual.input_shape, 
                kernels_per_layer=fittest_individual.kernels_per_layer, 
                kernel_sizes=fittest_individual.kernel_sizes,
                dense_units=fittest_individual.dense_units,
                dropout_probability=fittest_individual.dropout_probability,
                classes=fittest_individual.classes, 
                hyperparameters_filters=hyperparameters_filters
            )
            logger.debug("Model type: {}".format(type(model)))
            
            logger.debug("Compile the model.")
            model.compile(
                optimizer=Adam(learning_rate=fittest_individual.learning_rate[0]), 
                loss='binary_crossentropy', 
                metrics=['accuracy']
            )
            
            logger.debug("Return model.")
            return model

        logger.info("Set tuner.")
        tuner = keras_tuner.tuners.BayesianOptimization(
                model_builder,
                objective='val_accuracy',
                max_trials=50
        )

        logger.info("Start search.")
        tuner.search(
                self.population.x_train, 
                self.population.y_train,
                epochs=self.population.get_fittest().epochs,
                validation_data=(self.population.x_test, self.population.y_test)
        )
        logger.info("End search.")

        logger.info("Get the optimal hyperparameters.")
        logger.debug("tuner.get_best_hyperparameters type: {}.".format(type(tuner.get_best_hyperparameters)))
        logger.debug("tuner.get_best_hyperparameters len: {}.".format(len(tuner.get_best_hyperparameters)))
        best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
        best_model = tuner.hypermodel.build(best_hps)
        logger.debug("best_model type: {}.".format(type(best_model)))
        history = best_model.fit(
            self.population.x_train, 
            self.population.y_train,
            epochs=5,
            validation_data=(self.population.x_test, self.population.y_test)
        )

        logger.info("Evaluate model.")
        _, acc = best_model.evaluate(self.population.x_test, self.population.y_test, verbose=0)
        logger.info('> %.3f' % (acc * 100.0))
            
        return best_model
        # pass
