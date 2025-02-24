import pytest
from unittest.mock import patch
from unittest.mock import Mock

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

from src.models.supervised_learning.random_forest_classifier import RandomForestClassifier
from src.optimization.search_strategies.grid_search import GridSearchOptimizer
from src.models.templates.template_base_model import BaseModel


mock_logger = Mock()


class MockBaseModel(BaseModel):
    def fit(self, X, y):
        # Empty implementation for testing (optional)
        pass

    def evaluate(self, X, y) -> dict:
        # Empty implementation for testing
        return {}

    def predict(self, X) -> np.ndarray:
        # Empty implementation for testing
        return np.array([])


@pytest.fixture
def config():
    return {
        "config": {
            "param_grid": {"n_estimators": [10, 100]},
            "cv": 5,
            "scoring": "accuracy",
        }
    }


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_optimize_with_mock_model(config):
    # Create mock data
    mock_config = {
        "model": {
            "name": "random_forest_classifier",
            "config": {
                "n_estimators": 100
            }
        }
    }
    model = RandomForestClassifier(mock_config["model"], mock_logger).model
    # Create a pandas DataFrame with your data
    data = pd.DataFrame({
        "sepal length (cm)": [5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9, 5.4, 4.8, 4.8, 4.3, 5.8, 5.7, 5.4, 5.1, 5.7, 5.1, 5.4, 5.1, 4.6, 5.1, 4.8, 5.0, 5.0, 5.2, 5.2],
        "sepal width (cm)": [3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5, 3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3.0, 3.4, 3.5, 3.4],
        "petal length (cm)": [1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5, 1.5, 1.6, 1.4, 1.1, 1.2, 1.5, 1.3, 1.4, 1.7, 1.5, 1.7, 1.5, 1.0, 1.7, 1.9, 1.6, 1.6, 1.5, 1.4],
        "petal width (cm)": [0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2],
        "target": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })

    # Separate features and target
    X_train = data.iloc[:, :-1]  # Select columns excluding the last one (target)
    y_train = data.iloc[:, -1]  # Select the last column as the target
    
    # Create mock model and GridSearchOptimizer
    optimizer = GridSearchOptimizer(config, mock_logger)

    best_params, best_score = optimizer.optimize(model, X_train, y_train)

    # Assert best_params is a dictionary and best_score is not None
    assert isinstance(best_params, dict)
    assert best_score is not None


def test_optimize_raises_error_for_non_base_model(config):
    # Create GridSearchOptimizer
    optimizer = GridSearchOptimizer(config, mock_logger)

    # Test with a non-BaseModel object
    with pytest.raises(TypeError):
        optimizer.optimize("not_a_model", pd.DataFrame(), pd.Series())
