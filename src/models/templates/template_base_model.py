"""
This module defines an abstract base class
for machine learning models.

The `BaseModel` class provides a common interface for
different types of models, defining abstract methods for
training, making predictions, and evaluating model performance.
These methods must be implemented by any
concrete model class that inherits from `BaseModel`.

Classes:
    - BaseModel:
        Abstract base class for machine learning models.

Methods:
    - fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        Abstract method for training the model.
    - predict(self, X: pd.DataFrame) -> np.ndarray:
        Abstract method for making predictions.
    - evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        Abstract method for evaluating the model.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """
    Base class for machine learning models.

    This class provides a common interface for
    different types of models, defining abstract methods
    that must be implemented by subclasses.

    Attributes:
        config (dict):
            Configuration dictionary containing
            parameters for the model.
    """

    def __init__(
        self,
        config: dict
    ) -> None:
        """
        Initializes the BaseModel with a configuration dictionary.

        Args:
            config (dict):
                Configuration dictionary for the model.
        """
        self.config = config

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> None:
        """
        Trains the model on the given data.

        Args:
            X (pd.DataFrame):
                Input features.
            y (pd.Series):
                Target labels.
        """
        raise NotImplementedError(
            "Subclasses must implement the fit method."
        )

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Makes predictions on new data.

        Args:
            X (pd.DataFrame):
                Input features.

        Returns:
            np.ndarray:
                Predicted values.
        """
        raise NotImplementedError(
            "Subclasses must implement the predict method."
        )

    @abstractmethod
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> dict:
        """
        Evaluates the model's performance on the given data.

        Args:
            X (pd.DataFrame):
                Input features.
            y (pd.Series):
                True labels.

        Returns:
            dict:
                Dictionary containing evaluation metrics.
        """
        raise NotImplementedError(
            "Subclasses must implement the evaluate method."
        )
