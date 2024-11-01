"""
Factory class for creating various experiment components.

This class provides a centralized location for creating different types of
experiment components, including optimizers, data loaders, and models. It
uses a dictionary structure to map component types to their corresponding
classes, allowing for flexible and dynamic component creation.
"""
import logging

from src.optimization.search_strategies.grid_search import GridSearchOptimizer
from src.data.tabular_data import TabularDataLoader
from src.data.time_series_data import TimeSeriesDataLoader
from src.models.custom_random_forest_classifier import CustomRandomForestClassifier


class ComponentFactory:
    """
    Factory class for creating various experiment components.
    """

    def __init__(self):
        self.components = {
            "optimizer": {
                "grid_search": GridSearchOptimizer,
            },
            "data_loader": {
                "tabular_data": TabularDataLoader,
                "time_series_data": TimeSeriesDataLoader
            },
            "model": {
                "random_forest_classifier": CustomRandomForestClassifier,
            },
        }

    def create_component(
        self,
        component_type: str,
        component_name: str,
        component_config: dict,
        logger: logging.Logger
    ) -> object:
        """
        Creates an instance of the specified component
        based on name and configuration.

        Args:
            component_type (str):
                Type of component to create
                (optimizer, data_loader, model, etc.).
            component_name (str):
                Name of the specific component within the type.
            component_config (dict):
                Configuration dictionary for the component.
            logger (logging.Logger):
                An instance of the logger object shared in whole project.

        Returns:
            object:
                An instance of the chosen component class.

        Raises:
            ValueError:
                If the specified component or name is not supported.
        """

        component_specs = self.components.get(component_type.lower())
        if component_specs is None:
            raise ValueError(f"Unknown component type: {component_type}")

        component_class = component_specs.get(component_name.lower())
        if component_class is None:
            raise ValueError(f"Unknown {component_type}: {component_name}")

        return component_class(component_config, logger)
