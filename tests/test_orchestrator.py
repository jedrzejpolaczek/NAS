import pytest
from unittest.mock import patch
from src.orchestration.orchestrator import Orchestrator
from src.utils.component_factory import ComponentFactory
from src.experiment_management.experiment_tracker import ExperimentTracker


def test_init_with_valid_config():
    """Tests if Orchestrator initializes correctly with valid config."""
    config = {"log_dir": "tests/data/test_logs"}
    orchestrator = Orchestrator(config)
    assert orchestrator.config == config
    assert isinstance(orchestrator.component_factory, ComponentFactory)
    assert isinstance(orchestrator.experiment_tracker, ExperimentTracker)


@patch.object(ComponentFactory, "create_component")
def test_get_component(mock_create_component, orchestrator, component_type, component_config):
    """Tests if get_component delegates to ComponentFactory correctly."""
    mock_create_component.return_value = "component_instance"
    component = orchestrator.get_component(component_type, component_config)
    assert component == mock_create_component.return_value
    mock_create_component.assert_called_once_with(
        component_type=component_type, component_name=component_config["name"],
        component_config=component_config, logger=orchestrator.logger
    )


def test_run_experiment_raises_error_for_unknown_optimizer(orchestrator, experiment_config):
    """Tests if run_experiment raises ValueError for unknown optimizer."""
    experiment_config["optimizer"]["name"] = "unknown_optimizer"
    with pytest.raises(ValueError):
        orchestrator.run_experiment(experiment_config)


# Define fixtures for data and experiment config (optional)
@pytest.fixture
def component_type():
    return "optimizer"


@pytest.fixture
def component_config():
    return {"name": "grid_search"}


@pytest.fixture
def experiment_config():
    return {
            "optimizer": {
                "name": "grid_search",
                "config": {
                    "param_grid": {
                        "n_estimators": [10, 100],
                        "max_depth": [3, 5]
                    },
                    "cv": 3,
                    "scoring": "accuracy"
                }
            },
            "dataset": {
                "name": "tabular_data",
                "path": "data/raw/tabular_data",
                "file_name": "iris_dataset.csv",
                "dataset_name": "iris",
                "config": {
                    "test_size": 0.2,
                    "validation_size": 0.2
                }
            },
            "model": {
                "name": "random_forest_classifier",
                "config": {
                    "n_estimators": 100
                },
                "evaluation": {
                    "average": "micro"
                }
            }
        }

@pytest.fixture(autouse=True)
def orchestrator():
    config = {
    "log_dir": "logs",
    "experiments": [
            {
                "optimizer": {
                    "name": "grid_search",
                    "config": {
                        "param_grid": {
                            "n_estimators": [10, 100],
                            "max_depth": [3, 5]
                        },
                        "cv": 3,
                        "scoring": "accuracy"
                    }
                },
                "dataset": {
                    "name": "tabular_data",
                    "path": "data/raw/tabular_data",
                    "file_name": "iris_dataset.csv",
                    "dataset_name": "iris",
                    "config": {
                        "test_size": 0.2,
                        "validation_size": 0.2
                    }
                },
                "model": {
                    "name": "random_forest_classifier",
                    "config": {
                        "n_estimators": 100
                    },
                    "evaluation": {
                        "average": "micro"
                    }
                }
            }
        ]
    }
    return Orchestrator(config)
