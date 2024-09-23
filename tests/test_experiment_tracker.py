import json
import os
import pytest

from src.experiment_management.experiment_tracker import ExperimentTracker


def test_initialization_creates_directory(tmpdir):
    log_dir = str(tmpdir.join("log"))

    tracker = ExperimentTracker(log_dir)

    assert os.path.exists(log_dir)


def test_log_experiment(tmpdir):
    log_dir = str(tmpdir.join("log"))
    tracker = ExperimentTracker(log_dir)

    experiment_data = {"experiment_name": "experiment1", "accuracy": 0.95}
    tracker.log_experiment(experiment_data)

    with open(os.path.join(log_dir, 'experiments.json'), 'r') as f:
        logged_data = [json.loads(line) for line in f]

    assert logged_data == [experiment_data]
