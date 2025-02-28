import pytest
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from unittest.mock import (Mock, patch)

from src.models.ensemble_learning.random_forest_classifier import RandomForestClassifier


# Setup basic logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def rf_config():
    """Fixture providing a sample RandomForestClassifier configuration."""
    return {
        "config": {
            "n_estimators": 2,
            "max_depth": 3,
            "min_samples_split": 2
        },
        "evaluation": {
            "average": "binary"
        }
    }


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [2, 4, 6, 8]
    })
    y = pd.Series([0, 1, 0, 1])
    return X, y


def test_init(rf_config):
    """Test initialization of RandomForestClassifier."""
    rf = RandomForestClassifier(rf_config, logger)
    assert rf.n_estimators == 2
    assert rf.max_depth == 3
    assert rf.min_samples_split == 2
    assert rf.model is None
    assert rf.classes_ is None
    assert rf.config == rf_config
    assert rf.logger == logger


@patch('numpy.random.randint')
@patch('numpy.random.choice')
def test_fit(mock_choice, mock_randint, rf_config, sample_data):
    """Test fit method with mocked random operations."""
    X, y = sample_data
    rf = RandomForestClassifier(rf_config, logger)
    
    # Mock random operations for deterministic splits
    mock_choice.return_value = np.arange(len(X))  # No shuffling for simplicity
    mock_randint.return_value = 0  # Always split on feature1
    
    rf.fit(X, y)
    
    assert rf.model is not None
    assert len(rf.model) == 2  # n_estimators = 2
    assert isinstance(rf.model[0], dict)
    assert rf.classes_.tolist() == [0, 1]
    assert mock_choice.call_count == 2  # Once per tree


def test_to_tensor(rf_config):
    """Test _to_tensor method with different input types."""
    rf = RandomForestClassifier(rf_config, logger)
    
    # Test with DataFrame
    df = pd.DataFrame([[1, 2], [3, 4]])
    tensor_df = rf._to_tensor(df)
    assert isinstance(tensor_df, tf.Tensor)
    assert tensor_df.dtype == tf.float32
    
    # Test with Series
    series = pd.Series([0, 1])
    tensor_series = rf._to_tensor(series)
    assert isinstance(tensor_series, tf.Tensor)
    assert tensor_series.dtype == tf.int32
    
    # Test with NumPy array
    arr = np.array([[1, 2], [3, 4]])
    tensor_arr = rf._to_tensor(arr)
    assert isinstance(tensor_arr, tf.Tensor)
    assert tensor_arr.dtype == tf.float32
    
    # Test invalid input
    with pytest.raises(ValueError, match="Input data must be a pandas DataFrame, Series, or NumPy array"):
        rf._to_tensor("invalid")


@patch('numpy.random.randint')
@patch('numpy.random.choice')
def test_predict(mock_choice, mock_randint, rf_config, sample_data):
    """Test predict method after fitting."""
    X, y = sample_data
    rf = RandomForestClassifier(rf_config, logger)
    
    mock_choice.return_value = np.arange(len(X))
    mock_randint.return_value = 0
    
    rf.fit(X, y)
    predictions = rf.predict(X)
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert all(pred in [0, 1] for pred in predictions)


@patch('numpy.random.randint')
@patch('numpy.random.choice')
def test_predict_proba(mock_choice, mock_randint, rf_config, sample_data):
    """Test predict_proba method after fitting."""
    X, y = sample_data
    rf = RandomForestClassifier(rf_config, logger)
    
    mock_choice.return_value = np.arange(len(X))
    mock_randint.return_value = 0
    
    rf.fit(X, y)
    proba = rf.predict_proba(X)
    
    assert isinstance(proba, np.ndarray)
    assert proba.shape == (len(X), 2)  # 2 classes
    assert np.allclose(proba.sum(axis=1), 1)  # Probabilities sum to 1


@patch('numpy.random.randint')
@patch('numpy.random.choice')
def test_evaluate_binary(mock_choice, mock_randint, rf_config, sample_data):
    """Test evaluate method with binary classification."""
    X, y = sample_data
    rf = RandomForestClassifier(rf_config, logger)
    
    mock_choice.return_value = np.arange(len(X))
    mock_randint.return_value = 0
    
    rf.fit(X, y)
    metrics = rf.evaluate(X, y)
    
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0 <= metrics["accuracy"] <= 1


@patch('numpy.random.randint')
@patch('numpy.random.choice')
def test_evaluate_multiclass(mock_choice, mock_randint, sample_data):
    """Test evaluate method with multiclass (mocked config change)."""
    X, y = sample_data
    rf_config_multi = {
        "config": {"n_estimators": 2, "max_depth": 3, "min_samples_split": 2},
        "evaluation": {"average": "macro"}
    }
    rf = RandomForestClassifier(rf_config_multi, logger)
    
    mock_choice.return_value = np.arange(len(X))
    mock_randint.return_value = 0
    
    rf.fit(X, y)
    metrics = rf.evaluate(X, y)
    
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_get_params(rf_config):
    """Test get_params method."""
    rf = RandomForestClassifier(rf_config, logger)
    params = rf.get_params()
    
    assert isinstance(params, dict)
    assert params["n_estimators"] == 2
    assert params["max_depth"] == 3
    assert params["min_samples_split"] == 2
    assert params["config"] == rf_config
    assert params["logger"] == logger


def test_set_params(rf_config):
    """Test set_params method."""
    rf = RandomForestClassifier(rf_config, logger)
    
    new_params = {
        "n_estimators": 5,
        "max_depth": 7,
        "min_samples_split": 3,
        "logger": Mock()
    }
    rf.set_params(**new_params)
    
    assert rf.n_estimators == 5
    assert rf.max_depth == 7
    assert rf.min_samples_split == 3
    assert rf.logger == new_params["logger"]
    
    # Test setting config
    new_config = {
        "config": {"n_estimators": 10, "max_depth": 15},
        "evaluation": {"average": "binary"}
    }
    rf.set_params(config=new_config)
    assert rf.n_estimators == 10
    assert rf.max_depth == 15
    assert rf.config == new_config
