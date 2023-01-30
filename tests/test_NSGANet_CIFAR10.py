#!/usr/bin/env python
"""
Implementation of NSGANet on CIFAR10 data.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':
    import random
    import tensorflow
    from loguru import logger
    from sklearn.preprocessing import LabelBinarizer

    from NAS.search_space.populations import Population
    from NAS.search_space.binary_network_representation_with_skip_bit_individual import BinaryNetworkRepresentationWithSkipBitIndividual
    from NAS.search_strategies.genetic_algorithms.nsga_net.nsga_net import NSGANet

    # Setting log level
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tensorflow.keras.utils.disable_interactive_logging()
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Load Data
    cifar10 = tensorflow.keras.datasets.cifar10
    (train_images, train_labels), (_, _) = cifar10.load_data()

    y_train = train_labels.flatten()

    # Normalization and Reshaping
    input_shape = (32, 32, 3)

    x_train=train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 3)
    x_train=x_train / 255.0

    # Use only a subsample
    n = train_images.shape[0]
    lb = LabelBinarizer()
    lb.fit(range(10))
    selection = random.sample(range(n), 10000)  # Use only a subsample
    y_train = lb.transform(train_labels[selection])  # One-hot encodings
    x_train = train_images.reshape(n, 32, 32, 3)[selection]

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
