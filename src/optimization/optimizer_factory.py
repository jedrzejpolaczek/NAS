from src.optimization.search_strategies.grid_search import GridSearchOptimizer
from src.optimization.search_strategies.random_search import RandomSearchOptimizer


class OptimizerFactory:
    """
    Factory class for creating optimizer objects based on the specified name.
    """

    def __init__(self):
        self.optimizers = {
            "grid_search": GridSearchOptimizer,
            "random_search": RandomSearchOptimizer,
        }

    def create_optimizer(self, optimizer_name: str, optimizer_config: dict):
        """
        Creates an optimizer instance based on the name and configuration.

        Args:
            optimizer_name (str):
                The name of the optimizer to create.
            optimizer_config (dict):
                The configuration dictionary for the optimizer.

        Returns:
            OptimizerBase: An instance of the specified optimizer class.

        Raises:
            ValueError: If the specified optimizer is not supported.
        """

        optimizer_class = self.optimizers.get(optimizer_name.lower())
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer_class(optimizer_config)
