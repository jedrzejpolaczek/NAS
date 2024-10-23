"""
This module contains a custom implementation of the RandomForestClassifier.

It includes a custom class `CustomRandomForestClassifier`
that inherits from the `ClassificationModel` base class. 
The class provides specific implementations for fitting the model,
making predictions, predicting probabilities, 
and evaluating the model, using the Random Forest algorithm.

Classes:
    - CustomRandomForestClassifier:
        Inherits from `ClassificationModel` and
        provides a specific implementation 
        for the Random Forest algorithm.

Methods:
    - fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Fit the model to the training data.
    - predict(self, X: pd.DataFrame) -> np.ndarray:
        Predict the class labels for the provided data.
    - predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Return probability estimates for the test data.
    - evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        Evaluate the model's performance.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from src.models.templates.template_classification_model import ClassificationModel


class CustomRandomForestClassifier(ClassificationModel):
    """
    Custom Random Forest classifier model.

    Inherits from the `ClassificationModel` base class and
    provides a specific implementation
    for the Random Forest algorithm.
    """

    def __init__(
        self,
        config: dict,
        logger: logging.Logger
    ) -> None:
        """
        Initializes the CustomRandomForestClassifier with a configuration dictionary.

        Args:
            config (dict):
                Configuration dictionary for the model.
            logger (logging.Logger):
                An instance of the logger object shared in whole project.
        """
        super().__init__(config, logger)
        self.model = RandomForestClassifier(**self.config["config"])

    def fit(
        self,
        input_features: pd.DataFrame,
        target_labels: pd.Series
    ) -> None:
        """
        Trains the model on the given data.

        Args:
            input_features (pd.DataFrame):
                Input features. Commonly marked as X.
            target_labels (pd.Series):
                Target labels. Commonly marked as y.
        """
        self.model.fit(input_features, target_labels)

    def predict(
        self,
        input_features: pd.DataFrame
    ) -> np.ndarray:
        """
        Makes predictions on new data.

        Args:
            input_features (pd.DataFrame):
                Input features. Commonly marked as X.

        Returns:
            np.ndarray:
                Predicted class label per sample.
        """
        return self.model.predict(input_features)

    def predict_proba(
        self,
        input_features: pd.DataFrame
    ) -> np.ndarray:
        """
        Return probability estimates for the test data.

        Args:
            input_features (pd.DataFrame):
                Input features. Commonly marked as X.

        Returns:
            np.ndarray
                Returns the probability of the sample
                for each class in the model.
        """
        return self.model.predict_proba(input_features)

    def  evaluate(
        self,
        input_features: pd.DataFrame,
        target_labels: pd.Series
    ) -> dict:
        """
        Evaluates the model's performance on the given data.

        Args:
            input_features (pd.DataFrame):
                Input features. Commonly marked as X.
            target_labels (pd.Series):
                Target labels. Commonly marked as y.

        Returns:
            dict:
                Returns a dictionary containing metric scores.
        """
        target_labels_prediction = self.model.predict(input_features)
        return {
            "accuracy": accuracy_score(
                target_labels,
                target_labels_prediction
            ),
            "precision": precision_score(
                target_labels,
                target_labels_prediction,
                average=self.config["evaluation"]["average"]
            ),
            "recall": recall_score(
                target_labels,
                target_labels_prediction,
                average=self.config["evaluation"]["average"]
            ),
            "f1_score": f1_score(
                target_labels,
                target_labels_prediction,
                average=self.config["evaluation"]["average"]
            ),
        }

