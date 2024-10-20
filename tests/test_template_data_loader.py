# Import libraries
import os
import numpy as np
import pytest
from unittest.mock import Mock

from src.data.templates.template_data_loader import BaseDataLoader

# Mock logger for testing purposes
mock_logger = Mock()


class TestBaseDataLoader:

    def test_init_with_config(self):
        config = {"path": "data", "file_name": "data.csv"}
        loader = BaseDataLoader(config, mock_logger)
        assert loader.config == config
        assert loader.file_path == os.path.join(config["path"], config["file_name"])
        assert loader.data is None

    def test_init_without_logger(self):
        config = {"path": "data", "file_name": "data.csv"}
        with pytest.raises(TypeError):
            BaseDataLoader(config)

    @pytest.mark.skipif(True, reason="Abstract method, not implemented")
    def test_download_data(self):
        config = {"path": "data", "file_name": "data.csv"}
        loader = BaseDataLoader(config, mock_logger)
        with pytest.raises(NotImplementedError):
            loader.download_data()

    @pytest.mark.skipif(True, reason="Abstract method, not implemented")
    def test_load_data(self):
        config = {"path": "data", "file_name": "data.csv"}
        loader = BaseDataLoader(config, mock_logger)
        with pytest.raises(NotImplementedError):
            loader.load_data()

    @pytest.mark.skipif(True, reason="Abstract method, not implemented")
    def test_preprocess_data(self):
        config = {"path": "data", "file_name": "data.csv"}
        loader = BaseDataLoader(config, mock_logger)
        with pytest.raises(NotImplementedError):
            loader.preprocess_data()

    def test_get_data_empty(self):
        config = {"path": "data", "file_name": "data.csv"}
        loader = BaseDataLoader(config, mock_logger)
        assert loader.get_data() is None

    def test_get_data_with_data(self):
        config = {"path": "data", "file_name": "data.csv"}
        loader = BaseDataLoader(config, mock_logger)
        loader.data = {"X": [1, 2, 3], "y": ["a", "b", "c"]}
        assert loader.get_data() == loader.data

    def test_split_data(self, mocker):
        data = {"X": [1, 2, 3, 4], "y": ["a", "b", "c", "d"]}
        config = {
            "test_size": 0.25,
            "validation_size": 0.25,
            "path": "mocked_path",
            "file_name": "data.csv"
        }
        loader = BaseDataLoader(config, mock_logger)

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
