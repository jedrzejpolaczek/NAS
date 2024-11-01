import pytest
import json
from src.utils.config import load_config


def test_load_config_valid_path():
    """Tests if the function loads a valid JSON configuration file correctly."""
    config_path = "tests/data/test_config.json"  # Replace with the actual path to your test config file
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"key": "value"}, f)

    config = load_config(config_path)
    assert config == {"key": "value"}


def test_load_config_invalid_path():
    """Tests if the function raises an error for an invalid path."""
    config_path = "nonexistent_file.json"
    with pytest.raises(FileNotFoundError):
        load_config(config_path)


def test_load_config_invalid_json():
    """Tests if the function raises an error for an invalid JSON file."""
    config_path = "tests/data/invalid_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write("invalid json")

    with pytest.raises(json.decoder.JSONDecodeError):
        load_config(config_path)
