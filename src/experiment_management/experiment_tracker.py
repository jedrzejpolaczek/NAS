"""
Module to track and log experiment data.
"""
import json
import os
import numpy as np


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

    def _convert_to_json_serializable(self, data):
        """
        Recursively convert NumPy types to JSON-serializable Python types.

        Args:
            data: The data to convert (could be dict, list, numpy type, etc.).

        Returns:
            The converted data ready for JSON serialization.
        """
        if isinstance(data, dict):
            return {key: self._convert_to_json_serializable(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_to_json_serializable(item) for item in data]
        elif isinstance(data, np.integer):  # Handles numpy.int64, int32, etc.
            return int(data)
        elif isinstance(data, np.floating):  # Handles numpy.float64, float32, etc.
            return float(data)
        elif isinstance(data, np.ndarray):  # Handles NumPy arrays
            return self._convert_to_json_serializable(data.tolist())
        else:
            return data

    def log_experiment(self, experiment_data):
        """
        Log the experiment data to a JSON file.

        Args:
            experiment_data (dict): The data of the experiment to log.
        """
        log_experiment_path = os.path.join(self.log_dir, "experiments.json")

        # Convert experiment_data to ensure all values are JSON-serializable
        serializable_data = self._convert_to_json_serializable(experiment_data)

        with open(log_experiment_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(serializable_data) + "\n")

