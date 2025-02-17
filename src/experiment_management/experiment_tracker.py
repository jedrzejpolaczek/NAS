"""
Module to track and log experiment data.
"""
import json
import os


class ExperimentTracker:
    """
    Class for tracking and logging experiment data.

    Attributes:
        log_dir (str):
            The directory where the experiment data will be logged.
    """
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def log_experiment(self, experiment_data):
        """
        Log the experiment data.

        Args:
            experiment_data (dict):
                The data of the experiment to log.
        """
        log_experiment_path = os.path.join(self.log_dir, "experiments.json")

        with open(log_experiment_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(experiment_data) + "\n")
