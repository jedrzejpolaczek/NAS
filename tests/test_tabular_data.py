import pytest
from unittest.mock import Mock
import os
import pandas as pd

from src.data.tabular_data import TabularDataLoader

mock_logger = Mock()


@pytest.fixture
def config():
    return {"dataset_name": "iris", "path": "data/", "file_name": "iris.csv"}


def test_download_data_success(config):
    # Create a TabularDataLoader instance with the config
    loader = TabularDataLoader(config, mock_logger)

    # Call download_data and assert no exception is raised
    loader.download_data()

    # Check if the CSV file exists (optional)
    assert os.path.exists(loader.file_path)


def test_download_data_unsupported_dataset(config):
    # Modify the config to have an unsupported dataset name
    config["dataset_name"] = "not_supported"

    # Create a TabularDataLoader instance with the config
    loader = TabularDataLoader(config, mock_logger)

    # Call download_data and assert a ValueError is raised
    with pytest.raises(ValueError) as excinfo:
        loader.download_data()
    assert "Dataset not_supported" in str(excinfo.value)


def test_download_data_missing_dataset_name(config):
    # Modify the config to remove the dataset_name key
    del config["dataset_name"]

    # Create a TabularDataLoader instance with the config
    loader = TabularDataLoader(config, mock_logger)

    # Call download_data and assert a ValueError is raised
    with pytest.raises(AttributeError) as excinfo:
        loader.download_data()
    assert "Missing dataset name" in str(excinfo.value)


def test_load_data_success(config):
    # Mock the download_data method to avoid actually downloading
    loader = TabularDataLoader(config, mock_logger)
    loader.download_data = lambda: None

    # Load the data
    loader.load_data()

    # Assert data is loaded as a pandas DataFrame
    assert isinstance(loader.data, pd.DataFrame)


def test_load_data_missing_dataset_name(config):
    # Modify the config to remove the dataset_name key
    del config["dataset_name"]

    # Create a TabularDataLoader instance with the config
    loader = TabularDataLoader(config, mock_logger)

    # Call load_data and assert a ValueError is raised
    with pytest.raises(AttributeError) as excinfo:
        loader.load_data()
    assert "Missing dataset name" in str(excinfo.value)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_preprocess_data_success(config):
    # Mock the download_data and load_data methods to avoid actual download and loading
    loader = TabularDataLoader(config, mock_logger)
    loader.download_data = lambda: None
    loader.load_data = lambda: None
    loader.data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]})

    # Preprocess the data
    loader.preprocess_data()

    # Assert data is preprocessed (e.g., features scaled, missing values handled)
    # You can add specific assertions based on your preprocessing logic
    assert loader.data["X"].shape[1] == 2  # Check if features are scaled (number of columns)


def test_preprocess_data_no_data_loaded(config):
    # Create a TabularDataLoader instance with the config
    loader = TabularDataLoader(config, mock_logger)

    # Call preprocess_data and assert a ValueError is raised
    with pytest.raises(ValueError) as excinfo:
        loader.preprocess_data()
    assert "No data loaded" in str(excinfo.value)
