import pytest
import logging

from src.utils.component_factory import ComponentFactory
from src.optimization.search_strategies.grid_search import GridSearchOptimizer
from src.models.ensemble_learning.random_forest_classifier import RandomForestClassifier
from src.data.tabular_data import TabularDataLoader


def test_create_component_valid_optimizer():
    """Tests if create_component creates a GridSearchOptimizer for 'optimizer' type."""
    factory = ComponentFactory()
    component_config = {
        "config": {
            "param_grid": {
                "n_estimators": [10, 50],
                "max_depth": [3, 5],
                "min_samples_split": [2, 4]
            },
            "cv": 2,
            "scoring": "accuracy"
        }
    }
    logger = logging.getLogger("test_logger")
    component = factory.create_component(
        "optimizer", "grid_search", component_config, logger
    )
    assert isinstance(component, GridSearchOptimizer)
    assert component.config == component_config
    assert component.logger == logger


def test_create_component_valid_data_loader():
    """Tests if create_component creates a TabularDataLoader for 'data_loader' type."""
    factory = ComponentFactory()
    component_config = {
        "data_path": "data.csv",
        "name": "tabular_data",
        "path": "data/raw/tabular_data",
        "file_name": "iris_dataset.csv",
        "dataset_name": "iris",
        "config": {
            "test_size": 0.2,
            "validation_size": 0.2
        }
    }
    logger = logging.getLogger("test_logger")
    component = factory.create_component(
        "data_loader", "tabular_data", component_config, logger
    )
    assert isinstance(component, TabularDataLoader)
    assert component.config == component_config
    assert component.logger == logger


def test_create_component_valid_model():
    """Tests if create_component creates a RandomForestClassifier for 'model' type."""
    factory = ComponentFactory()
    model_config = {"config":{"n_estimators": 100}}  # Configuration for the model
    logger = logging.getLogger("test_logger")
    component = factory.create_component(
        "model", "random_forest_classifier", model_config, logger
    )
    assert isinstance(component, RandomForestClassifier)
    assert component.config == model_config  # Check only model config
    assert component.logger == logger


def test_create_component_unknown_type():
    """Tests if create_component raises ValueError for unknown component type."""
    factory = ComponentFactory()
    component_config = {}
    logger = logging.getLogger("test_logger")
    with pytest.raises(ValueError) as excinfo:
        factory.create_component("unknown_type", "name", component_config, logger)
    assert str(excinfo.value) == "Unknown component type: unknown_type"


def test_create_component_unknown_name():
    """Tests if create_component raises ValueError for unknown name within a type."""
    factory = ComponentFactory()
    component_config = {}
    logger = logging.getLogger("test_logger")
    with pytest.raises(ValueError) as excinfo:
        factory.create_component("optimizer", "unknown_name", component_config, logger)
    assert str(excinfo.value) == "Unknown optimizer: unknown_name"
