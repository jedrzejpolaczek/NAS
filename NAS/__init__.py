from loguru import logger

# Make public APIs available at top-level import
from NAS.search_space.populations import Population

# Genetic Algorithms
try:
    from NAS.search_strategies.genetic_algorithms.genetic_algorithm import GeneticAlgorithm 
    from NAS.search_strategies.genetic_algorithms.nsga_2.nsga_2 import NSGA2
    from NAS.search_strategies.genetic_algorithms.nsga_net.nsga_net import NSGANet
except ImportError:
    logger.warning("Warning: install search startegy to use GeneticAlgorithm, NSGA2 and NSGANet.")

# Binary string network representation with skip bit individuals and models
try:
    from NAS.search_space.binary_string_network_representation_with_skip_bit_individual import BinaryStringNetworkRepresentationWithSkipBitIndividual
    from NAS.models.binary_string_network_representation_with_skip_bit_model import BinaryStringNetworkRepresentationWithSkipBitModel
except ImportError:
    logger.warning("Warning: install search space and models to use \
        BinaryStringNetworkRepresentationWithSkipBitIndividual and BinaryStringNetworkRepresentationWithSkipBitModel.")

# Bayesian optimization algorithm
try:
    from NAS.search_strategies.bayesian_optimization import bayesian_optimization_algorithm
except ImportError:
    logger.warning("Warning: install search startegy to use bayesian_optimization_algorithm.")