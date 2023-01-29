#!/usr/bin/env python
"""
Implementation of Genetic CNN on MNIST data.
This is a replica of the algorithm described
on section 4.1.1 of the Genetic CNN paper.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    import mnist
    import random

    import os

    import tensorflow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tensorflow.keras.utils.disable_interactive_logging()
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    from sklearn.preprocessing import LabelBinarizer
    from NAS.search_space.populations import Population
    from NAS.search_space.binary_network_representation_with_skip_bit_individual import BinaryNetworkRepresentationWithSkipBitIndividual
    from NAS.search_strategies.genetic_algorithms.nsga_net.nsga_net import NSGANet

    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    n = train_images.shape[0]
    lb = LabelBinarizer()
    lb.fit(range(10))
    selection = random.sample(range(n), 10000)  # Use only a subsample
    y_train = lb.transform(train_labels[selection])  # One-hot encodings
    x_train = train_images.reshape(n, 28, 28, 1)[selection]
    x_train = x_train / 255  # Normalize train data

    population = Population(
        BinaryNetworkRepresentationWithSkipBitIndividual, 
        x_train, 
        y_train, 
        size=20, 
        crossover_rate=0.3, 
        mutation_rate=0.1,
        additional_parameters={
            'kfold': 2, 
            'epochs': (3, 1), 
            'learning_rate': (1e-3, 1e-4), 
            'batch_size': 32
        }, 
        maximize=True
    )
    ga = NSGANet(population, crossover_probability=0.2, mutation_probability=0.8)

    ga.run(max_generations=5)
