import pytest
from unittest.mock import Mock
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from src.models.supervised_learning.random_forest_classifier import RandomForestClassifier


mock_logger = Mock()


@pytest.fixture
def config():
    return {
        "config":{
            "n_estimators": 100,
            "max_depth": 5
        },
        "evaluation": {
            "average": "micro"
        }
    }


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_fit_predict(config):
    # Create a random dataset
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    # Create a RandomForestClassifier instance
    model = RandomForestClassifier(config, mock_logger)

    # Fit the model
    model.fit(pd.DataFrame(X), pd.Series(y))

    # Predict
    predictions = model.predict(pd.DataFrame(X))

    # Assert predictions are not all the same class
    assert not np.all(predictions == predictions[0])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_predict_proba(config):
    # Create a random dataset
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    # Create a RandomForestClassifier instance
    model = RandomForestClassifier(config, mock_logger)

    # Fit the model
    model.fit(pd.DataFrame(X), pd.Series(y))

    # Predict probabilities
    probabilities = model.predict_proba(pd.DataFrame(X))

    # Assert probabilities sum to 1 for each sample
    assert np.allclose(probabilities.sum(axis=1), np.ones(len(X)))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_evaluate(config):
    # Create a random dataset
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    # Create a RandomForestClassifier instance
    model = RandomForestClassifier(config, mock_logger)

    # Fit the model
    model.fit(pd.DataFrame(X), pd.Series(y))

    # Evaluate the model
    metrics = model.evaluate(pd.DataFrame(X), pd.Series(y))

    # Assert metrics are within a reasonable range (adjust as needed)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
