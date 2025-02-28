# Import libraries
import os
import pytest
from unittest.mock import Mock

from src.data.templates.template_data_loader import BaseDataLoader


# Concrete subclass for testing
class ConcreteDataLoader(BaseDataLoader):
    """A concrete subclass of BaseDataLoader for testing purposes."""
    def download_data(self) -> None:
        """Simple implementation that logs a message and does nothing."""
        self.logger.info("Downloading data...")


    def load_data(self) -> None:
        """Simple implementation that sets mock data."""
        self.logger.info("Loading data...")
        self.data = {"X": [1, 2, 3, 4], "y": ["a", "b", "c", "d"]}


    def preprocess_data(self) -> None:
        """Simple implementation that logs a message and modifies data."""
        self.logger.info("Preprocessing data...")
        if self.data:
            self.data["X"] = [x * 2 for x in self.data["X"]]  # Double the X values


# Fixture for mock logger to ensure fresh instance per test
@pytest.fixture
def mock_logger():
    return Mock()


class TestBaseDataLoader:
    """Test class for BaseDataLoader using a concrete subclass"""
    def test_init_with_config(self, mock_logger):
        config = {"path": "data", "file_name": "data.csv"}
        loader = ConcreteDataLoader(config, mock_logger)
        assert loader.config == config
        assert loader.file_path == os.path.join(config["path"], config["file_name"])
        assert loader.data is None


    def test_init_without_logger(self):
        config = {"path": "data", "file_name": "data.csv"}
        with pytest.raises(TypeError):
            BaseDataLoader(config)


    def test_download_data(self, mock_logger):
        config = {"path": "data", "file_name": "data.csv"}
        loader = ConcreteDataLoader(config, mock_logger)
        loader.download_data()
        mock_logger.info.assert_called_once_with("Downloading data...")
        assert loader.data is None  # download_data doesn’t set data


    def test_load_data(self, mock_logger):
        config = {"path": "data", "file_name": "data.csv"}
        loader = ConcreteDataLoader(config, mock_logger)
        loader.load_data()
        mock_logger.info.assert_called_once_with("Loading data...")
        assert loader.data == {"X": [1, 2, 3, 4], "y": ["a", "b", "c", "d"]}


    def test_preprocess_data(self, mock_logger):
        config = {"path": "data", "file_name": "data.csv"}
        loader = ConcreteDataLoader(config, mock_logger)
        loader.data = {"X": [1, 2, 3, 4], "y": ["a", "b", "c", "d"]}
        loader.preprocess_data()
        mock_logger.info.assert_called_once_with("Preprocessing data...")
        assert loader.data == {"X": [2, 4, 6, 8], "y": ["a", "b", "c", "d"]}


    def test_get_data_empty(self, mock_logger):
        config = {"path": "data", "file_name": "data.csv"}
        loader = ConcreteDataLoader(config, mock_logger)
        assert loader.get_data() is None


    def test_get_data_with_data(self, mock_logger):
        config = {"path": "data", "file_name": "data.csv"}
        loader = ConcreteDataLoader(config, mock_logger)
        loader.data = {"X": [1, 2, 3], "y": ["a", "b", "c"]}
        assert loader.get_data() == loader.data


    def test_split_data(self, mock_logger, mocker):
        data = {"X": [1, 2, 3, 4], "y": ["a", "b", "c", "d"]}
        config = {
            "test_size": 0.25,
            "validation_size": 0.25,
            "path": "mocked_path",
            "file_name": "data.csv"
        }
        loader = ConcreteDataLoader(config, mock_logger)

        # Mock the file_path attribute
        mocker.patch.object(loader, "file_path", "mocked_path")

        x_train, x_val, x_test, y_train, _, _ = loader.split_data(data)

        # Assert lengths based on split sizes
        assert len(x_train) == int(0.5 * len(data["X"]))
        assert len(x_val) == int(0.25 * len(data["X"]))
        assert len(x_test) == int(0.25 * len(data["X"]))

        # Assert type of returned values (should be arrays)
        assert isinstance(x_train, list)
        assert isinstance(y_train, list)
