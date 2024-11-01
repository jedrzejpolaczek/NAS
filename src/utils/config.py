"""
This module provides a function to load a JSON configuration file.

Functions:
    load_config(config_path): Loads a JSON configuration file from the specified path.
"""
import json


def load_config(
    config_path: str
) -> dict:
    """
    Loads a JSON configuration file from the specified path.

    Args:
        config_path (str):
            The path to the JSON configuration file.

    Returns:
        dict:
            The loaded JSON configuration as a dictionary.
    """

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
