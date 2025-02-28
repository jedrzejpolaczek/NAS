import pytest
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from unittest.mock import (Mock, patch)

from src.optimization.search_strategies.random_search import RandomSearchOptimizer
from src.models.templates.template_base_model import BaseModel


# Mock BaseModel for testing
class MockBaseModel(BaseModel):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.params = {}

    def set_params(self, **params):
        self.params.update(params)

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        # Simulate predictions: return 1 if param 'a' > 5, else 0
        return np.ones(len(X)) if self.params.get('a', 0) > 5 else np.zeros(len(X))

    def evaluate(self, X, y):
        # Dummy implementation for abstract method 'evaluate'
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# Fixture for logger
@pytest.fixture
def logger():
    return logging.getLogger("test_logger")


# Fixture for sample config
@pytest.fixture
def sample_config():
    return {
        "config": {
            "param_distributions": {
                "a": [1, 5, 10],
                "b": [0.1, 0.5, 1.0]
            },
            "n_iter": 3,
            "cv": 2,
            "scoring": "accuracy"
        }
    }


# Fixture for sample data
@pytest.fixture
def sample_data():
    X = pd.DataFrame({"feature1": [1, 2, 3, 4], "feature2": [5, 6, 7, 8]})
    y = pd.Series([0, 1, 0, 1])
    return X, y


# Test initialization
def test_random_search_optimizer_init(sample_config, logger):
    optimizer = RandomSearchOptimizer(sample_config, logger)
    assert optimizer.param_distributions == sample_config["config"]["param_distributions"]
    assert optimizer.n_iter == 3
    assert optimizer.cv == 2
    assert optimizer.scoring == "accuracy"


# Test _generate_random_combinations
def test_generate_random_combinations(sample_config, logger):
    optimizer = RandomSearchOptimizer(sample_config, logger)
    combinations = optimizer._generate_random_combinations()
    
    assert len(combinations) == 3  # Matches n_iter
    for combo in combinations:
        assert "a" in combo and combo["a"] in [1, 5, 10]
        assert "b" in combo and combo["b"] in [0.1, 0.5, 1.0]


# Test _cross_validate with mocking
@patch('numpy.random.shuffle')  # Mock shuffle to ensure predictable splits
def test_cross_validate(mock_shuffle, sample_config, logger, sample_data):
    mock_shuffle.side_effect = lambda x: None  # No shuffling for deterministic test
    optimizer = RandomSearchOptimizer(sample_config, logger)
    model = MockBaseModel({"some": "config"}, logger)
    X, y = sample_data
    params = {"a": 10, "b": 0.5}  # 'a' > 5 should yield accuracy 1

    score = optimizer._cross_validate(model, params, X, y)
    
    assert isinstance(score, float)
    assert 0 <= score <= 1
    # With params a=10, predictions should be all 1s, matching half of y (0,1,0,1)
    assert score == pytest.approx(0.5, abs=0.1)  # Depends on split, approximate check


# Test _cross_validate unsupported scoring
def test_cross_validate_unsupported_scoring(sample_config, logger, sample_data):
    sample_config["config"]["scoring"] = "unsupported"
    optimizer = RandomSearchOptimizer(sample_config, logger)
    model = MockBaseModel({"some": "config"}, logger)
    X, y = sample_data
    
    with pytest.raises(ValueError, match="Scoring method 'unsupported' not supported yet."):
        optimizer._cross_validate(model, {"a": 1}, X, y)


# Test optimize method
@patch.object(RandomSearchOptimizer, '_generate_random_combinations')
@patch.object(RandomSearchOptimizer, '_cross_validate')
def test_optimize(mock_cv, mock_combinations, sample_config, sample_data):
    # Use a Mock logger instead of the fixture
    mock_logger = Mock()
    optimizer = RandomSearchOptimizer(sample_config, mock_logger)
    model = MockBaseModel({"some": "config"}, mock_logger)
    X, y = sample_data
    
    # Mock return values
    mock_combinations.return_value = [
        {"a": 1, "b": 0.1},
        {"a": 10, "b": 0.5}
    ]
    mock_cv.side_effect = [0.3, 0.7]  # Scores for the two param sets
    
    best_params, best_score = optimizer.optimize(model, X, y)
    
    assert best_params == {"a": 10, "b": 0.5}  # Higher score
    assert best_score == 0.7
    assert mock_cv.call_count == 2  # Called for each combination
    # Check that the completion message was logged (among others)
    assert mock_logger.info.call_count == 2  # Two info calls: start and complete
    mock_logger.info.assert_any_call("Random Search optimization completed.")


# Test edge case: empty param_distributions
def test_empty_param_distributions(logger):
    config = {
        "config": {
            "param_distributions": {},
            "n_iter": 1,
            "cv": 2,
            "scoring": "accuracy"
        }
    }
    optimizer = RandomSearchOptimizer(config, logger)
    combinations = optimizer._generate_random_combinations()
    assert combinations == [{}]  # Should return one empty dict


# Test edge case: n_iter = 0
def test_zero_n_iter(sample_config, logger):
    sample_config["config"]["n_iter"] = 0
    optimizer = RandomSearchOptimizer(sample_config, logger)
    combinations = optimizer._generate_random_combinations()
    assert combinations == []  # Should return empty list
