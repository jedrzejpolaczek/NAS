import pytest
import logging
import pandas as pd
import numpy as np

from src.optimization.search_strategies.grid_search import GridSearchOptimizer
from src.models.templates.template_base_model import BaseModel


# Setup basic logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock BaseModel class for testing
class MockModel(BaseModel):
    """A mock implementation of BaseModel for testing purposes."""
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.params = {}

    def fit(self, X, y):
        """Mock fit method; does nothing."""
        pass

    def predict(self, X):
        """Mock predict method; returns array of ones matching input length."""
        return np.ones(len(X))

    def evaluate(self, X, y):
        """Mock evaluate method; returns a dummy dict to satisfy ABC."""
        return {"dummy_metric": 0.0}

    def set_params(self, **params):
        """Mock set_params; updates internal params dict."""
        self.params.update(params)
        return self

    def get_params(self, deep=True):
        """Mock get_params; returns current params."""
        return self.params


@pytest.fixture
def grid_search_config():
    """Fixture providing a sample GridSearchOptimizer configuration."""
    return {
        "config": {
            "param_grid": {"param1": [1, 2], "param2": [3, 4]},
            "cv": 3,
            "scoring": "accuracy"
        }
    }


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    X = pd.DataFrame({"feature1": [1, 2, 3, 4, 5, 6], "feature2": [2, 4, 6, 8, 10, 12]})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    return X, y


def test_init(grid_search_config):
    """Test initialization of GridSearchOptimizer."""
    optimizer = GridSearchOptimizer(grid_search_config, logger)
