"""
This module provides a function to load a JSON configuration file.

Functions:
    load_config(config_path):
        Loads a JSON configuration file from the specified path.
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
    if not isinstance(config_path, str):
        raise TypeError("config_path must be a string.")

    if not config_path.endswith(".json"):
        raise ValueError("config_path must point to a JSON file.")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object")

        return config

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        ) from e

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in configuration file: {str(e)}", e.doc, e.pos
        ) from e
