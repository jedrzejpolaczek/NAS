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
from abc import (ABC, abstractmethod)
import logging
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
        config: dict,
        logger: logging.Logger
    ) -> None:
        """
        Initializes the BaseModel with a configuration dictionary.

        Args:
            config (dict):
                Configuration dictionary for the model.
            logger (logging.Logger):
                An instance of the logger object shared in whole project.
        """
        self.config = config
        self.logger = logger

    @abstractmethod
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
        raise NotImplementedError(
            "Subclasses must implement the fit method."
        )

    @abstractmethod
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
                Predicted values.
        """
        raise NotImplementedError(
            "Subclasses must implement the predict method."
        )

    @abstractmethod
    def evaluate(
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
                Dictionary containing evaluation metrics.
        """
        raise NotImplementedError(
            "Subclasses must implement the evaluate method."
        )
