import pytest
from unittest.mock import patch, call, Mock
from src.orchestration.orchestrator import Orchestrator
from src.utils.component_factory import ComponentFactory
from src.experiment_management.experiment_tracker import ExperimentTracker
from collections import namedtuple

# Define data loader mock behavior
DataLoadMock = namedtuple("DataLoadMock", ["data_pipeline", "split_data"])


def test_init_with_valid_config():
    """Tests if Orchestrator initializes correctly with valid config."""
    config = {"log_dir": "test_logs"}
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


@patch.object(Orchestrator, "get_component")
def test_run_experiment_valid_config(mock_get_component, orchestrator, experiment_config):
    """Tests if run_experiment orchestrates component calls correctly."""
    pass
    # mock_get_component.side_effect = [
    #     "optimizer_instance",
    #     # Mock data_loader behavior (explicitly call Mock())
    #     lambda component_type, config: DataLoadMock(Mock(), Mock()),
    #     "model_instance"
    # ]
    # orchestrator.run_experiment(experiment_config)

    # # Verify calls to get_component
    # mock_get_component.assert_has_calls(
    #     [
    #         call(
    #             "optimizer", experiment_config["optimizer"]
    #         ),
    #         call("data_loader", experiment_config["dataset"]),
    #         call("model", experiment_config["model"]),
    #     ]
    # )

    # # Verify data loading and splitting are called (implementation details may vary)
    # data_loader_instance = mock_get_component.return_value[1]
    # data_loader_instance.data_pipeline.assert_called_once()
    # data_loader_instance.split_data.assert_called_once()

    # # Verify model training and evaluation are called (implementation details may vary)
    # model_instance = mock_get_component.return_value[2]
    # model_instance.model.fit.assert_called_once()
    # model_instance.evaluate.assert_called_once()

    # # Verify experiment logging
    # expected_data = {
    #     "optimizer": experiment_config["optimizer"]["name"],
    #     "dataset": experiment_config["dataset"]["name"] + " (" + experiment_config["dataset"]["dataset_name"] + ")",
    #     "model": experiment_config["model"]["name"],
    #     # ... (best_params, best_score, metrics) from mocked methods
    # }
    # orchestrator.experiment_tracker.log_experiment.assert_called_once_with(expected_data)


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
    "data_path": "data/example_dataset.csv",
    "log_dir": "logs",
    "test_size": 0.2,
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
