"""
This module contains a custom implementation of the RandomForestClassifier.

It includes a custom class `CustomRandomForestClassifier`
that inherits from the `ClassificationModel` base class. 
The class provides specific implementations for fitting the model,
making predictions, predicting probabilities, 
and evaluating the model, using the Random Forest algorithm.

Classes:
    - CustomRandomForestClassifier:
        Inherits from `ClassificationModel` and provides a specific implementation 
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

    Inherits from the `ClassificationModel` base class and provides a specific implementation
    for the Random Forest algorithm.
    """

    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)
        self.model = RandomForestClassifier(**self.config)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """
        Fit the model to the training data.

        Args:
            X (pd.DataFrame):
                Training input samples.
            y (pd.DataFrame):
                Target values.
        """
        self.model.fit(X, y)

    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Args:
            X (pd.DataFrame)
                Test input samples.

        Returns:
            np.ndarray:
                Predicted class label per sample.
        """
        return self.model.predict(X)

    def predict_proba(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Return probability estimates for the test data.

        Args:
            X (pd.DataFrame):
                Test input samples.

        Returns:
            np.ndarray
                Returns the probability of the sample
                for each class in the model.
        """
        return self.model.predict_proba(X)

    def  evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> dict:
        """
        Evaluate the model's performance.

        Args:
            X (pd.DataFrame):
                Test input samples.
            y (pd.Series):
                True labels for test set.

        Returns:
            dict:
                Returns a dictionary containing metric scores.
        """
        y_pred = self.model.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),  
            "f1_score": f1_score(y, y_pred),  
        }
