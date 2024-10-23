"""
This module defines a base class for classification models.

The `ClassificationModel` class inherits from the `BaseModel` class
and serves as a base for all classification models.
It adds the `predict_proba` method, which should be implemented by
any concrete classification model class
that inherits from `ClassificationModel`.

Classes:
    - ClassificationModel:
        Base class for classification models. Inherits from `BaseModel`.

Methods:
    - predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Abstract method for predicting class probabilities.
"""
from abc import abstractmethod
import pandas as pd
import numpy as np

from src.models.templates.template_base_model import BaseModel


class ClassificationModel(BaseModel):
    """
    Base class for classification models.

    Inherits from `BaseModel` and provides additional
    methods specific to classification tasks.
    """

    @abstractmethod
    def predict_proba(
        self,
        input_features: pd.DataFrame
    ) -> np.ndarray:
        """
        Predicts class probabilities for new data.

        Args:
            input_features (pd.DataFrame):
                Input features. Commonly marked as X.

        Returns:
            np.ndarray:
                Predicted class probabilities.
        """
        raise NotImplementedError(
            "Subclasses must implement the predict_proba method."
        )
