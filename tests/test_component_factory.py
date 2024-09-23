import pytest

from src.optimization.search_strategies.grid_search import GridSearchOptimizer
from src.data.tabular_data import TabularDataLoader
from src.models.custom_random_forest_classifier import CustomRandomForestClassifier
from src.utils.component_factory import ComponentFactory


def test_create_component_unknow_name():
    """"Test that ValueError is raised for unknown component name."""
    component_factory = ComponentFactory()

    with pytest.raises(ValueError):
        component_factory.create_component("model", "unknown_name", {})


def test_create_component_unknown_type():
    """Test that ValueError is raised for unknown component type."""
    component_factory = ComponentFactory()
    
    with pytest.raises(ValueError):
        component_factory.create_component("unknown_type", "unknown_name", {})


def test_create_component_grid_search():
    """Test the creation of a GridSearchOptimizer."""
    component_factory = ComponentFactory()

    config = {"config": {"param_grid": {}, "cv": 5, "scoring": "accuracy"}}
    component = component_factory.create_component("optimizer", "grid_search", config)
    assert isinstance(component, GridSearchOptimizer)


def test_create_component_tabular_data():
    """Test the creation of a TabularDataLoader."""
    component_factory = ComponentFactory()
    
    component = component_factory.create_component("data_loader", "tabular_data", {})
    assert isinstance(component, TabularDataLoader)


def test_create_component_random_forest():
    """Test the creation of a CustomRandomForestClassifier."""
    component_factory = ComponentFactory()
    
    config = {"n_estimators": 100, "max_depth": None}
    component = component_factory.create_component("model", "random_forest_classifier", config)
    assert isinstance(component, CustomRandomForestClassifier)
