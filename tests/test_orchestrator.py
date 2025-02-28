import pytest
from unittest.mock import (Mock, patch, call)

from src.orchestration.orchestrator import Orchestrator
from src.utils.component_factory import ComponentFactory


# Fixture for sample config
@pytest.fixture
def sample_config():
    return {
        "log_dir": "/mock/log/dir",
        "experiments": [
            {
                "experiment_name": "exp1",
                "optimizer": {
                    "name": "random_search",
                    "config": {
                        "param_distributions": {"a": [1, 2], "b": [0.1, 0.2]},
                        "n_iter": 10,
                        "cv": 5,
                        "scoring": "accuracy"
                    }
                },
                "model": {"name": "mock_model", "param2": "value2"},
                "dataset": {"name": "mock_dataset", "dataset_name": "test_data", "param3": "value3"}
            }
        ]
    }


# Fixture for mock logger
@pytest.fixture
def mock_logger():
    return Mock()


# Test initialization
@patch('src.orchestration.orchestrator.ExperimentTracker')
def test_orchestrator_init(mock_experiment_tracker_class, sample_config, mock_logger):
    mock_tracker_instance = Mock()
    mock_tracker_instance.log_dir = "/mock/log/dir"
    mock_experiment_tracker_class.return_value = mock_tracker_instance
    orchestrator = Orchestrator(sample_config, mock_logger)
    
    assert orchestrator.config == sample_config
    assert isinstance(orchestrator.component_factory, ComponentFactory)
    assert orchestrator.experiment_tracker == mock_tracker_instance
    assert orchestrator.experiment_tracker.log_dir == "/mock/log/dir"
    assert orchestrator.logger == mock_logger
    mock_experiment_tracker_class.assert_called_once_with("/mock/log/dir")

# Test get_component
@patch('src.orchestration.orchestrator.ExperimentTracker')
@patch.object(ComponentFactory, 'create_component')
def test_get_component(mock_create_component, mock_experiment_tracker_class, sample_config, mock_logger):
    mock_tracker_instance = Mock()
    mock_tracker_instance.log_dir = "/mock/log/dir"
    mock_experiment_tracker_class.return_value = mock_tracker_instance
    orchestrator = Orchestrator(sample_config, mock_logger)
    component_config = {"name": "test_component", "param": "value"}
    
    component = orchestrator.get_component("test_type", component_config)
    
    mock_create_component.assert_called_once_with(
        component_type="test_type",
        component_name="test_component",
        component_config=component_config,
        logger=mock_logger
    )
    assert component == mock_create_component.return_value


# Test run_experiment (unchanged)
@patch('src.orchestration.orchestrator.ExperimentTracker')
@patch.object(ComponentFactory, 'create_component')
def test_run_experiment(mock_create_component, mock_experiment_tracker_class, sample_config, mock_logger):
    mock_tracker_instance = Mock()
    mock_tracker_instance.log_dir = "/mock/log/dir"
    mock_experiment_tracker_class.return_value = mock_tracker_instance
    orchestrator = Orchestrator(sample_config, mock_logger)
    experiment_config = sample_config["experiments"][0]
    
    mock_data_loader = Mock()
    mock_data_loader.data_pipeline = Mock()
    mock_data_loader.get_data = Mock(return_value={"data": "mocked"})
    mock_data_loader.split_data = Mock(return_value=("input_train", "input_val", "input_test", "target_train", "target_val", "target_test"))
    
    mock_model = Mock()
    mock_model.set_params = Mock()
    mock_model.fit = Mock()
    mock_model.evaluate = Mock(return_value={"accuracy": 0.9})
    
    mock_optimizer = Mock()
    mock_optimizer.optimize = Mock(return_value=({"param1": 1}, 0.85))
    
    mock_create_component.side_effect = [mock_optimizer, mock_model, mock_data_loader]
    
    with patch.object(mock_tracker_instance, 'log_experiment') as mock_log_experiment:
        orchestrator.run_experiment(experiment_config)
    
    assert mock_create_component.call_count == 3
    expected_calls = [
        call(component_type="optimizer", component_name="random_search", component_config=experiment_config["optimizer"], logger=mock_logger),
        call(component_type="model", component_name="mock_model", component_config=experiment_config["model"], logger=mock_logger),
        call(component_type="data_loader", component_name="mock_dataset", component_config=experiment_config["dataset"], logger=mock_logger)
    ]
    mock_create_component.assert_has_calls(expected_calls, any_order=False)
    
    mock_data_loader.get_data.assert_called_once()
    mock_data_loader.split_data.assert_called_once_with({"data": "mocked"})
    mock_optimizer.optimize.assert_called_once()
    mock_model.set_params.assert_called_once_with(param1=1)
    mock_model.fit.assert_called_once()
    mock_model.evaluate.assert_called_once()
    mock_log_experiment.assert_called_once()


# Test run method with successful logger setup
@patch('src.orchestration.orchestrator.ExperimentTracker')
@patch('src.orchestration.orchestrator.get_logger')
@patch('src.orchestration.orchestrator.Orchestrator.run_experiment')
def test_run_success(mock_run_experiment, mock_get_logger, mock_experiment_tracker_class, sample_config, mock_logger):
    mock_tracker_instance = Mock()
    mock_tracker_instance.log_dir = "/mock/log/dir"
    mock_experiment_tracker_class.return_value = mock_tracker_instance
    mock_new_logger = Mock()
    mock_get_logger.return_value = mock_new_logger
    orchestrator = Orchestrator(sample_config, mock_logger)
    
    print("Before run()")
    orchestrator.run()
    print("After run()")
    
    print("get_logger call count:", mock_get_logger.call_count)
    print("get_logger calls:", mock_get_logger.call_args_list)
    print("run_experiment calls:", mock_run_experiment.call_args_list)
    
    mock_get_logger.assert_called_once_with(
        name="exp1",
        log_file="/mock/log/dir/exp1.log"
    )
    mock_run_experiment.assert_called_once_with(sample_config["experiments"][0])
    assert mock_logger.info.call_count == 1
    mock_logger.info.assert_called_once_with("Set logger configuration for experiment exp1...")
    mock_new_logger.info.assert_any_call("Running experiment: exp1")
    mock_new_logger.info.assert_any_call("Experiment exp1 completed.")
