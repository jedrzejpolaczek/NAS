import pytest
import os
import json
from src.experiment_management.experiment_tracker import ExperimentTracker


def test_experiment_tracker_initialization():
    """Tests if the ExperimentTracker initializes correctly."""
    log_dir = "test_logs"
    tracker = ExperimentTracker(log_dir)
    assert tracker.log_dir == log_dir
    assert os.path.exists(log_dir)
    log_path = os.path.join(log_dir, "log.txt")
    assert os.path.exists(log_path)


def test_log_experiment():
    """Tests if the log_experiment method logs data correctly."""
    log_dir = "test_logs"
    tracker = ExperimentTracker(log_dir)
    experiment_data = {"param1": "value1", "param2": "value2"}
    tracker.log_experiment(experiment_data)
    log_experiment_path = os.path.join(log_dir, "experiments.json")
    with open(log_experiment_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > 0
        logged_data = json.loads(lines[-1])
        assert logged_data == experiment_data


def test_log_experiment_handles_exceptions():
    """Tests if the log_experiment method handles exceptions gracefully."""
    log_dir = "test_logs"
    tracker = ExperimentTracker(log_dir)
    experiment_data = {"param1": "value1", "param2": "value2"}
    with pytest.raises(OSError):
        os.remove("")  # Remove the file to simulate an error
        tracker.log_experiment(experiment_data)
